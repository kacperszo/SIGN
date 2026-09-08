"""Score and embed a directory of complexes with the trained SIGN, in one pass.

`train.py` and `prepare.py` between them cover training but leave no way to run the model on
new complexes, which is what the harness needs. This closes that: build the graphs, load the
checkpoint, and write both outputs from a single forward.

    prediction   pred_score, the affinity scalar
    embedding    graph_feat, the 128-dimensional pooled representation the head reads

The embedding is native — `forward` already returns it, so nothing is pooled or invented here.
The head is `output_layer`; everything before it is the encoder.

The model's other output, the interaction matrix over 36 atom-pair types, is an auxiliary
training target and is not written. It is a property of the complex rather than a
representation of it, and the probe has nothing to do with it.

usage:
    python -m sign_torch.predict --complexes /data --model ckpt.pt --out /outputs
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .model import SIGN
from .prepare import _build_one
from .train import collate


def load_model(path: Path, device: str) -> tuple[SIGN, int]:
    """Rebuild the network from the sizes the checkpoint carries, not from defaults.

    `train.py` stores `infeat_dim`, `hidden_dim`, `num_convs` and `num_angle` beside the
    weights. Guessing any of them would either fail to load or, worse, load into a network
    shaped differently from the one that was trained. `num_angle` comes back out because the
    graph builder needs it too — the angle domains are part of the input, not just the model.
    """
    # weights_only: this is a plain state dict plus four integers, so there is no reason to
    # open it with the interpreter's full powers available
    ckpt = torch.load(path, map_location=device, weights_only=True)
    # `dense_dims` was not recorded before train.py started writing it, and the value it
    # trained with is not model.py's default — (512, 256, 128) against (128, 128, 64). A
    # checkpoint from before that change loads only with this fallback, and taking the
    # default instead fails loudly on the head's shapes rather than quietly.
    dense_dims = tuple(ckpt.get("dense_dims") or (512, 256, 128))
    model = SIGN(infeat_dim=ckpt["infeat_dim"], hidden_dim=ckpt["hidden_dim"],
                 num_convs=ckpt["num_convs"], dense_dims=dense_dims,
                 num_angle=ckpt["num_angle"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, ckpt["num_angle"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--complexes", required=True, type=Path,
                   help="directory of <id>/ holding <id>_protein.pdb and <id>_ligand.sdf")
    p.add_argument("--model", required=True, type=Path, help="checkpoint from train.py")
    p.add_argument("--out", required=True, type=Path, help="directory for the results")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cut_dist", type=float, default=5.0,
                   help="must match what the checkpoint was trained with")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=0,
                   help="parallel graph building; 0 runs serially")
    args = p.parse_args()

    model, num_angle = load_model(args.model, args.device)

    targets = sorted(d.name for d in args.complexes.iterdir() if d.is_dir())
    jobs = [(cid, str(args.complexes / cid), None, args.cut_dist, num_angle)
            for cid in targets]

    graphs, failed = [], []
    if args.workers and args.workers > 1:
        import multiprocessing as mp

        with mp.Pool(args.workers) as pool:
            results = pool.imap_unordered(_build_one, jobs, chunksize=4)
            for cid, data, error in results:
                (failed if error else graphs).append(f"{cid}: {error}" if error else data)
    else:
        for job in jobs:
            cid, data, error = _build_one(job)
            (failed if error else graphs).append(f"{cid}: {error}" if error else data)

    if not graphs:
        raise SystemExit("no graphs built")
    # graph building can come back out of order under a pool, and the ids have to travel with
    # the vectors rather than with the input listing
    graphs.sort(key=lambda g: g.complex_id)

    ids, scores, vectors = [], [], []
    with torch.no_grad():
        for start in range(0, len(graphs), args.batch_size):
            chunk = graphs[start:start + args.batch_size]
            batch = collate(chunk)
            for name in vars(batch):
                value = getattr(batch, name)
                if torch.is_tensor(value):
                    setattr(batch, name, value.to(args.device))
                elif isinstance(value, list) and value and torch.is_tensor(value[0]):
                    setattr(batch, name, [v.to(args.device) for v in value])
            _inter, score, graph_feat = model(batch)
            ids.extend(g.complex_id for g in chunk)
            scores.extend(score.reshape(-1).cpu().tolist())
            vectors.append(graph_feat.cpu().numpy())

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["complex_id", "y_pred"])
        writer.writerows(zip(ids, scores))
    np.savez(args.out / "embeddings.npz", ids=np.array(ids),
             vectors=np.concatenate(vectors, axis=0))

    print(f"scored {len(ids)} of {len(targets)} complexes")
    for line in failed:
        print(f"  skipped {line}")
    print(f"-> {args.out / 'predictions.csv'}")
    print(f"-> {args.out / 'embeddings.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
