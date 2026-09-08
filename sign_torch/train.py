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

usage:
    python -m sign_torch.train --graphs graphs.pt --out model.pt --epochs 50
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .model import SIGN


def collate(graphs: list) -> object:
    """Merge complexes into one batch, offsetting each graph type by its own node count."""
    from types import SimpleNamespace

    x, batch_idx = [], []
    a2a, dist, b2a = [], [], []
    b2b = [[] for _ in range(len(graphs[0].b2b_edge_index_list))]
    bond_types, type_counts, ys = [], [], []
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
        y=torch.cat(ys) if all(v is not None for v in ys) else None,
        num_graphs=len(graphs),
    )


def interaction_target(type_count: torch.Tensor) -> torch.Tensor:
    """The auxiliary label: the observed bond-type distribution per complex."""
    counts = type_count.t().float()
    total = counts.sum(dim=1, keepdim=True).clamp(min=1.0)
    return counts / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SIGN")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--val_graphs", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda_", type=float, default=1.75)
    parser.add_argument("--dec_step", type=int, default=8000)
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
                 num_angle=args.num_angle).to(device)
    optimiser = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=args.dec_step, gamma=args.lr_dec_rate)

    def run_epoch(graphs, training: bool):
        model.train(training)
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
            with torch.set_grad_enabled(training):
                inter_hat, score, _feat = model(batch)
                target = interaction_target(batch.type_count)
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

    best = float("inf")
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        mae, mae_inter, rmse, pearson = run_epoch(train, training=True)
        line = (f"epoch {epoch:3d}  train MAE {mae:6.3f}  inter {mae_inter:6.3f}  "
                f"RMSE {rmse:6.3f}  R {pearson:6.3f}")
        if val:
            v_mae, _v_inter, v_rmse, v_pearson = run_epoch(val, training=False)
            line += f"  |  val MAE {v_mae:6.3f}  RMSE {v_rmse:6.3f}  R {v_pearson:6.3f}"
            score_for_best = v_rmse
        else:
            score_for_best = rmse
        print(line, flush=True)

        if score_for_best < best:
            best = score_for_best
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "score": best,
                        "infeat_dim": train[0].x.shape[1],
                        "hidden_dim": args.hidden_dim,
                        "num_convs": args.num_convs,
                        "dense_dims": list(dense_dims),
                        "num_angle": args.num_angle}, args.out)

    print(f"\nbest {best:.4f} -> {args.out}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
