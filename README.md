# SIGN — structure-aware interaction graph network, rewritten in PyTorch

> A fork maintained for [gnn-benchmark](../../README.md). The authors' own README is
> kept as [README.upstream.md](README.upstream.md) for attribution and for their
> description of the method — **its build and run instructions are not current for
> this fork.**

## What it is

Atoms and bonds update in alternation, with bond-to-bond attention split across
angular domains, over three graph types: atom-to-atom at a 5 A cutoff whose *edges* are what the model
calls bonds, bond-to-atom with exactly one edge per bond, and bond-to-bond grouped by the angle
between them. Two outputs trained jointly — an affinity scalar and an auxiliary interaction matrix
over 36 ligand/pocket atom-pair types.

The original is PaddlePaddle + PGL and ships no checkpoints, so this is a rewrite rather than a port.
`preprocess_pdbbind.py` and `featurizer.py` are the authors' own and unchanged, so the featurisation
is theirs even though the graphs are not.

## State

| | |
|---|---|
| CASF-2016 scoring | **R 0.725**, RMSE 1.511, n=285 |
| CASF-2016 ranking | **rho 0.653** — best measured here; size baseline 0.619 |
| embedding | 128d, **native**, probe R 0.711, retains 98% of its own head |
| `gnnb verify` | 285/285, 8.9e-16 |
| training | 4285 complexes, 30 epochs, 23 min on an RTX 4060 Ti |

**The only model here whose training data is known**, and its split excludes the CASF-2016 core set
explicitly. No checkpoint ships with this fork — `*.pt` is gitignored deliberately, because the
authors publish no weights either.

## Build

```bash
podman build --format=docker -f Containerfile.torch -t sign:latest .     # cpu, tests only
podman build --format=docker -f Containerfile.gpu   -t sign-gpu:latest . # what training needs
```

## Run it, without the harness

Generated from this model's adapter by `gnnb howto`, so these are the exact commands
the benchmark issues — regenerate with `python tools/sync_model_readmes.py`. Every one
runs with `--network=none` and a read-only root filesystem.

Input is one directory per complex:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2; several models try both

Needs `<id>_pocket.pdb` and `<id>_ligand.mol2`. **Nothing ships `<id>_pocket.mol2`** — zero of 285 CASF-2016 complexes and zero of PDBbind v2019 — so `prepare.py` converts the pdb with Open Babel on the way in.

```bash
# sign.torch — localhost/sign:latest
# source: models/sign

# predict
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD:/ckpt:ro" \
    localhost/sign:latest \
    sh -c 'cd /work && TMPDIR=/tmp python -m sign_torch.predict --complexes /data --model /ckpt/checkpoint_trained.pt --out /outputs --device cpu'

# embed
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD:/ckpt:ro" \
    localhost/sign:latest \
    sh -c 'cd /work && TMPDIR=/tmp python -m sign_torch.predict --complexes /data --model /ckpt/checkpoint_trained.pt --out /outputs --device cpu'

# train
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD:/ckpt:ro" \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/sign:latest \
    sh -c 'cd /work && if [ -f /cache/sign_train_cut5_ang6.pt ]; then echo reusing /cache/sign_train_cut5_ang6.pt; else TMPDIR=/tmp python -m sign_torch.prepare --complexes /data --labels /splits/train.csv --out /cache/sign_train_cut5_ang6.pt --cut_dist 5 --num_angle 6; fi && if [ -f /cache/sign_val_cut5_ang6.pt ]; then echo reusing /cache/sign_val_cut5_ang6.pt; else TMPDIR=/tmp python -m sign_torch.prepare --complexes /data --labels /splits/val.csv --out /cache/sign_val_cut5_ang6.pt --cut_dist 5 --num_angle 6; fi && python -m sign_torch.train --graphs /cache/sign_train_cut5_ang6.pt --out /outputs --seed 0 --val_graphs /cache/sign_val_cut5_ang6.pt --epochs 30 --device cpu'

# finetune  (encoder frozen; drop --freeze-encoder to tune all of it)
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD:/ckpt:ro" \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/sign:latest \
    sh -c 'cd /work && if [ -f /cache/sign_train_cut5_ang6.pt ]; then echo reusing /cache/sign_train_cut5_ang6.pt; else TMPDIR=/tmp python -m sign_torch.prepare --complexes /data --labels /splits/train.csv --out /cache/sign_train_cut5_ang6.pt --cut_dist 5 --num_angle 6; fi && if [ -f /cache/sign_val_cut5_ang6.pt ]; then echo reusing /cache/sign_val_cut5_ang6.pt; else TMPDIR=/tmp python -m sign_torch.prepare --complexes /data --labels /splits/val.csv --out /cache/sign_val_cut5_ang6.pt --cut_dist 5 --num_angle 6; fi && python -m sign_torch.train --graphs /cache/sign_train_cut5_ang6.pt --out /outputs --seed 0 --val_graphs /cache/sign_val_cut5_ang6.pt --epochs 30 --init-encoder /ckpt/encoder.pt --freeze-encoder --device cpu'
```

## What comes out

| file | holds |
|---|---|
| `predictions.csv` | `complex_id,y_pred` |
| `embeddings.npz` | `ids` and `vectors`, 128-dim, native — `forward` returns it |
| `model.pt` | training only; reloads under `weights_only=True` |
| `history.csv` | training only; `epoch,train_loss,train_r,val_loss,val_r` and SIGN's own columns |
| `summary.json` | training only; best epoch, and what the run started from |

## Before you trust the numbers

**No checkpoint ships with this fork.** `.gitignore` excludes `*.pt` on purpose, so a fresh clone has to train one — which is the honest position, because the authors publish no weights either. That is also why this is the only model here whose training data is known: the split is ours and it excludes the CASF-2016 core set explicitly.

The pocket conversion is worth suspicion rather than comfort: pdb to mol2 assigns bond orders by perception, and Open Babel warns *\"Failed to kekulize aromatic bonds\"* while doing it. First place to look if a retrained model underperforms for no other visible reason.

## Maintainer notes

`CLAUDE.md` in this directory holds what breaks if it is changed back.
