"""Train SIGN. Hyperparameters read from the authors' train.py rather than chosen here.

    lambda_ 1.75   weight on the auxiliary interaction-matrix loss
    lr 0.001, Adam, halved every dec_step optimiser steps
    batch 32, hidden 128, num_convs 2, num_heads 4, feat_drop 0.2, dense 512/256/128

Both losses are L1 with **sum** reduction, as theirs are — with mean reduction the auxiliary
term would be scaled differently against the affinity term and lambda_ would no longer mean
what they tuned it to mean.

Batching is done by hand rather than with PyG's DataLoader. SIGN carries three graph types
whose indices refer to different node sets — atoms, bonds, and bonds again per angle domain —
and PyG's collation only knows how to offset `edge_index` against `num_nodes`. Getting the
bond offsets wrong would not raise; it would quietly wire one complex's bonds to another's.

**Fine-tuning.** `--init-encoder` loads a representation somebody else trained and attaches a
fresh head; `--freeze-encoder` trains the head alone. The boundary and the reasons for it are in
`transfer.py`. The outputs follow the harness contract in `harness/training.py` — `model.pt`,
`history.csv` and `summary.json` — so a transfer curve can be put beside another model's without
either trainer knowing about the other.

usage:
    python -m sign_torch.train --graphs graphs.pt --out /outputs --epochs 50
    python -m sign_torch.train --graphs graphs.pt --out /outputs \
        --init-encoder /ckpt/encoder.pt --freeze-encoder
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .model import SIGN
from .transfer import set_training_mode, transfer_encoder


def collate(graphs: list) -> object:
    """Merge complexes into one batch, offsetting each graph type by its own node count."""
    from types import SimpleNamespace

    x, batch_idx = [], []
    a2a, dist, b2a = [], [], []
    b2b = [[] for _ in range(len(graphs[0].b2b_edge_index_list))]
    bond_types, type_counts, type_pairs, ys = [], [], [], []
    atom_offset = bond_offset = 0

    for i, g in enumerate(graphs):
        x.append(g.x)
        batch_idx.append(torch.full((g.num_nodes,), i, dtype=torch.long))
        a2a.append(g.a2a_edge_index + atom_offset)
        dist.append(g.a2a_dist)
        # b2a spans two node sets: bonds on the source side, atoms on the destination
        b2a.append(g.b2a_edge_index + torch.tensor([[bond_offset], [atom_offset]]))
        for k, e in enumerate(g.b2b_edge_index_list):
            b2b[k].append(e + bond_offset)
        bond_types.append(g.bond_types)
        type_counts.append(g.type_count.reshape(-1, 1))
        type_pairs.append(getattr(g, "type_pair", None))
        # `hasattr` is not enough: prepare.py sets `y` to None when no label file was given,
        # so an unlabelled graph has the attribute and it holds nothing. Scoring new
        # complexes is exactly that case, and torch.cat on a None fails several frames away
        # from the cause.
        ys.append(getattr(g, "y", None))
        atom_offset += g.num_nodes
        bond_offset += g.num_bonds

    return SimpleNamespace(
        x=torch.cat(x),
        batch=torch.cat(batch_idx),
        a2a_edge_index=torch.cat(a2a, dim=1),
        a2a_dist=torch.cat(dist),
        b2a_edge_index=torch.cat(b2a, dim=1),
        b2b_edge_index_list=[torch.cat(e, dim=1) for e in b2b],
        bond_types=torch.cat(bond_types),
        type_count=torch.cat(type_counts, dim=1),
        # (B, 36), or None when any graph lacks it — scoring does not need it, training does
        type_pair=(torch.stack([t.reshape(-1) for t in type_pairs])
                   if all(t is not None for t in type_pairs) else None),
        y=torch.cat(ys) if all(v is not None for v in ys) else None,
        num_graphs=len(graphs),
    )


def interaction_target(batch) -> torch.Tensor:
    """The auxiliary label, as the authors define it: (B, 36), each row summing to one.

    The counts are `prepare.authors_interaction_counts` — pairs within 12 A over the whole
    protein — normalised exactly as their dataset.py:91 does, `inter_feats / inter_feats.sum()`.

    This used to be built from `type_count`, the 36 types counted over the graph's 5 A edges.
    That is a different distribution and was never the authors' target; see the docstring of
    `authors_interaction_counts`. A cache prepared before the fix carries no `type_pair` and
    is refused rather than silently falling back.
    """
    if batch.type_pair is None:
        raise SystemExit(
            "these graphs carry no `type_pair`, the authors' 12 A interaction target. They were "
            "prepared before 2026-09-24, when the port trained against a 5 A substitute. "
            "Delete the cached graph file and prepare again.")
    counts = batch.type_pair.float()
    return counts / counts.sum(dim=1, keepdim=True).clamp(min=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SIGN")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--val_graphs", default=None)
    parser.add_argument("--out", required=True,
                        help="output directory; model.pt, history.csv and summary.json are "
                             "written into it. A path ending in .pt is taken as the checkpoint "
                             "itself and the other two go beside it, so older invocations keep "
                             "working.")
    parser.add_argument("--init-encoder", dest="init_encoder", default=None,
                        help="encoder .pt from `gnnb encoder split` — start from a transferred "
                             "representation with a fresh head")
    parser.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true",
                        help="train the heads only, with the encoder in eval mode so its "
                             "dropout does not resample the representation each step")
    # The authors' train.py default. This was 50 until 2026-09-24, and the checkpoint behind
    # every SIGN number so far was trained for 30 — a tenth of the schedule the paper used.
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_", type=float, default=1.75)
    parser.add_argument("--dec_step", type=int, default=8000)
    parser.add_argument("--state", default=None,
                        help="training-state file, rewritten after every epoch. If it exists at "
                             "start, training resumes from it — model, optimiser, learning-rate "
                             "schedule, epoch count, best model, history and RNG state. Put it "
                             "under /cache so it outlives the run directory")
    parser.add_argument("--max-minutes", dest="max_minutes", type=float, default=None,
                        help="stop cleanly at the first epoch boundary after this many minutes "
                             "of this invocation, leaving --state ready to resume")
    parser.add_argument("--lr_dec_rate", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_convs", type=int, default=2)
    parser.add_argument("--num_angle", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda",
                        help="falls back to cpu if no GPU is visible, but CPU is not viable "
                             "here: one epoch over 4285 complexes does not finish in minutes")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device.startswith("cuda"):
        print("WARNING: no GPU visible; falling back to CPU, which will take hours")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    train = torch.load(args.graphs, weights_only=False)
    val = torch.load(args.val_graphs, weights_only=False) if args.val_graphs else None
    print(f"{len(train)} training graphs" + (f", {len(val)} validation" if val else ""))

    # The authors' head widths, and wider than model.py's default of (128, 128, 64). Named
    # here so it can be written into the checkpoint below: a saved network that does not
    # record every size needed to rebuild it can only be reloaded by someone who already
    # knows what it was.
    dense_dims = (512, 256, 128)
    model = SIGN(infeat_dim=train[0].x.shape[1], hidden_dim=args.hidden_dim,
                 num_convs=args.num_convs, dense_dims=dense_dims,
                 num_angle=args.num_angle)

    provenance: dict = {"init_encoder": None, "frozen_params": 0}
    if args.init_encoder:
        provenance = transfer_encoder(model, args.init_encoder, args.freeze_encoder)
    elif args.freeze_encoder:
        raise SystemExit("--freeze-encoder without --init-encoder would train the head against "
                         "a random representation; that is not an experiment")
    model.to(device)

    # Only the parameters that still want gradients. Handing Adam frozen tensors is not merely
    # wasteful: it keeps optimiser state for them and makes the "trainable parameters" line in
    # any log a lie.
    trainable = [q for q in model.parameters() if q.requires_grad]
    optimiser = Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=args.dec_step, gamma=args.lr_dec_rate)

    def run_epoch(graphs, training: bool):
        set_training_mode(model, training, frozen_encoder=args.freeze_encoder)
        order = torch.randperm(len(graphs)) if training else torch.arange(len(graphs))
        total, total_inter, n = 0.0, 0.0, 0
        preds, trues = [], []

        for start in range(0, len(order), args.batch_size):
            chunk = [graphs[i] for i in order[start:start + args.batch_size]]
            batch = collate(chunk)
            # Batches move to the GPU one at a time: the full training set is 3 GB and
            # holding it on the card buys nothing when each step touches one batch.
            for field in ("x", "batch", "a2a_edge_index", "a2a_dist", "b2a_edge_index",
                          "bond_types", "type_count", "y"):
                setattr(batch, field, getattr(batch, field).to(device, non_blocking=True))
            batch.b2b_edge_index_list = [e.to(device, non_blocking=True)
                                         for e in batch.b2b_edge_index_list]
            if batch.type_pair is not None:
                batch.type_pair = batch.type_pair.to(device, non_blocking=True)
            with torch.set_grad_enabled(training):
                inter_hat, score, _feat = model(batch)
                target = interaction_target(batch)
                loss = F.l1_loss(score.squeeze(-1), batch.y, reduction="sum")
                loss_inter = F.l1_loss(inter_hat, target, reduction="sum")
                objective = loss + args.lambda_ * loss_inter

            if training:
                optimiser.zero_grad()
                objective.backward()
                optimiser.step()
                scheduler.step()

            total += float(loss)
            total_inter += float(loss_inter)
            n += len(chunk)
            preds.append(score.squeeze(-1).detach().cpu())
            trues.append(batch.y.cpu())

        preds, trues = torch.cat(preds), torch.cat(trues)
        rmse = float(((preds - trues) ** 2).mean().sqrt())
        if len(preds) > 1 and preds.std() > 0:
            centred_p, centred_t = preds - preds.mean(), trues - trues.mean()
            pearson = float((centred_p * centred_t).sum()
                            / (centred_p.norm() * centred_t.norm()).clamp(min=1e-8))
        else:
            pearson = float("nan")
        return total / n, total_inter / n, rmse, pearson

    # `--out` is a directory under the harness contract, but a .pt path is what every earlier
    # invocation passed, so both are accepted and the three artefacts stay together either way.
    out = Path(args.out)
    if out.suffix == ".pt":
        model_path, out_dir = out, out.parent
    else:
        model_path, out_dir = out / "model.pt", out
    out_dir.mkdir(parents=True, exist_ok=True)

    best, best_epoch = float("inf"), -1
    best_state = None
    history: list[dict] = []
    started = time.time()
    first_epoch, earlier_seconds, resumed_from = 1, 0.0, None

    # **Resuming.** A long schedule does not have to fit in one sitting: the whole training
    # state is written after every epoch and read back here. Saved and loaded with plain
    # torch.save / weights_only=True — tensors, numbers, strings and lists of them, nothing that
    # needs unpickling code. Numpy's RNG is not in it because nothing here draws from numpy:
    # batch order is torch.randperm and dropout is torch's own.
    if args.state and Path(args.state).exists():
        state = torch.load(args.state, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        if torch.cuda.is_available() and state.get("rng_cuda"):
            torch.cuda.set_rng_state_all([t.cpu() for t in state["rng_cuda"]])
        first_epoch = int(state["epoch"]) + 1
        best, best_epoch = float(state["best"]), int(state["best_epoch"])
        best_state = state["best_model"]
        history = list(state["history"])
        earlier_seconds = float(state["seconds"])
        resumed_from = args.state
        print(f"resumed from {args.state} after epoch {first_epoch - 1} "
              f"(best {best:.4f} at epoch {best_epoch})", flush=True)

    stopped_for_time = False
    for epoch in range(first_epoch, args.epochs + 1):
        mae, mae_inter, rmse, pearson = run_epoch(train, training=True)
        line = (f"epoch {epoch:3d}  train MAE {mae:6.3f}  inter {mae_inter:6.3f}  "
                f"RMSE {rmse:6.3f}  R {pearson:6.3f}")
        # The first five keys are the harness's standard history columns; the rest are SIGN's
        # own and are carried because losing the auxiliary term would hide half the objective.
        row = {"epoch": epoch, "train_loss": mae, "train_r": pearson,
               "val_loss": "", "val_r": "",
               "train_rmse": rmse, "train_inter": mae_inter, "val_rmse": ""}
        if val:
            v_mae, _v_inter, v_rmse, v_pearson = run_epoch(val, training=False)
            line += f"  |  val MAE {v_mae:6.3f}  RMSE {v_rmse:6.3f}  R {v_pearson:6.3f}"
            row.update(val_loss=v_mae, val_r=v_pearson, val_rmse=v_rmse)
            score_for_best = v_rmse
        else:
            score_for_best = rmse
        history.append(row)
        print(line, flush=True)

        if score_for_best < best:
            best, best_epoch = score_for_best, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "score": best,
                        "infeat_dim": train[0].x.shape[1],
                        "hidden_dim": args.hidden_dim,
                        "num_convs": args.num_convs,
                        "dense_dims": list(dense_dims),
                        "num_angle": args.num_angle}, model_path)

        if args.state:
            # Written to a sibling and renamed, so an interruption mid-write leaves the
            # previous epoch's state intact rather than a truncated file.
            tmp = Path(args.state).with_suffix(".tmp")
            torch.save({
                "model": model.state_dict(), "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "best": best, "best_epoch": best_epoch, "best_model": best_state,
                "history": history,
                "seconds": earlier_seconds + (time.time() - started),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else []),
            }, tmp)
            tmp.replace(args.state)

        if args.max_minutes is not None and time.time() - started >= args.max_minutes * 60:
            stopped_for_time = epoch < args.epochs
            if stopped_for_time:
                print(f"time budget of {args.max_minutes:g} min reached after epoch {epoch}; "
                      f"resume with the same --state", flush=True)
            break

    # model.pt is the best model so far, written here as well as on each improvement: a run that
    # resumed and never improved would otherwise leave its own run directory without one.
    if best_state is not None:
        torch.save({"model_state_dict": best_state,
                    "epoch": best_epoch, "score": best,
                    "infeat_dim": train[0].x.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "num_convs": args.num_convs,
                    "dense_dims": list(dense_dims),
                    "num_angle": args.num_angle}, model_path)

    with (out_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (out_dir / "summary.json").write_text(json.dumps({
        "model": "sign",
        "epochs": len(history),
        "best_epoch": best_epoch,
        "best_score": best,
        "selected_on": "val_rmse" if val else "train_rmse",
        "train_graphs": len(train),
        "val_graphs": len(val) if val else 0,
        "seed": args.seed,
        "freeze_encoder": args.freeze_encoder,
        "seconds": round(earlier_seconds + time.time() - started, 1),
        "epochs_requested": args.epochs,
        # False means the schedule is unfinished: resume with the same --state.
        "complete": not stopped_for_time,
        "resumed_from": resumed_from,
        "state": args.state,
        **provenance,
    }, indent=2) + "\n")

    print(f"\nbest {best:.4f} at epoch {best_epoch} -> {model_path}  "
          f"({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
