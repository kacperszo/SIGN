<!-- gnn-benchmark:begin -->
# Running this in gnn-benchmark

**Rewritten from PaddlePaddle + PGL to PyTorch here** — the Paddle path is abandoned. SIGN
ships no checkpoints, so training was required either way and there was nothing to verify a
faithful port against; running the original would have bought a second framework and a second
GPU stack for one model, and bought nothing in fidelity. `preprocess_pdbbind.py` and
`featurizer.py` contain no Paddle and are reused unchanged, so the featurisation is theirs.

Trained and evaluated: CASF-2016 **R 0.725**, ranking **rho 0.653** — the best ranking of every
model measured — and a 128-dimensional native embedding that retains 98% of its own head.

| variant | capabilities | `gnnb verify` on CASF-2016 |
|---|---|---|
| `sign.torch` | predict, embed, train, finetune | 285/285, max abs diff 8.9e-16 |

The golden is this checkpoint's own recorded output. SIGN publishes no weights, so there is
nothing external to be faithful to and `verify` is a regression check on the port and its
environment rather than a fidelity claim.

```bash
podman build --format=docker -f Containerfile.torch -t sign:latest .        # cpu
podman build --format=docker -f Containerfile.gpu   -t sign-gpu:latest .

gnnb verify --variant sign.torch --dataset data/CASF-2016/coreset
gnnb run --variant sign.torch --capability predict --dataset <complexes> --gpu
gnnb run --variant sign.torch --capability embed   --dataset <complexes>
```

It is also the first model wired to the training contract, so it is the worked example for the
rest:

```bash
gnnb encoder split --variant sign.torch --out /tmp/sign_encoder.pt

gnnb train --variant sign.torch --dataset data/pdbbind_v2019 \
    --train-split benchmarks/pdbbind_train.csv --val-split benchmarks/pdbbind_val.csv \
    --epochs 30 --workers 10 --cache ~/.cache/gnnb --gpu

gnnb train --variant sign.torch --dataset data/pdbbind_v2019 \
    --train-split benchmarks/pdbbind_train.csv --val-split benchmarks/pdbbind_val.csv \
    --init-encoder /tmp/sign_encoder.pt --freeze-encoder --gpu
```

The head is `output_layer.output_layer` and `pipool_layer` — **not** the whole `output_layer`
module, which produces the 128-dim embedding one line before the affinity Linear reads it.
`sign_torch/test_transfer.py` pins that: after transfer the model must embed identically and
score differently.

**The trained checkpoint is not in git** — `.gitignore` excludes `*.pt` deliberately, and the
registry bind-mounts it from the working tree. A fresh clone has to train it or be handed the
file.

It is the only model in the benchmark whose training data is known, and its split excludes the
CASF-2016 core set explicitly. 266 of the 285 core-set complexes sit inside PDBbind refined and
no other repository here publishes a training list, so SIGN is being judged more strictly than
everything beside it — report its numbers with that attached. Full commands and the caveat in
[CLAUDE.md](CLAUDE.md).

## Running it without the harness

This fork runs on its own; the benchmark adds bookkeeping, not capability. Every
command below is generated from the adapter by `gnnb howto`, so it cannot drift from
what the harness actually runs — regenerate with `python tools/sync_model_readmes.py`.

All of them run with `--network=none` and a read-only root filesystem. Nothing is
fetched at run time; dependencies are resolved when the image is built.

### What it eats

One directory per complex, named after it:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2; several models try both

Needs `<id>_pocket.pdb` and `<id>_ligand.mol2`. **Nothing ships `<id>_pocket.mol2`** — zero of 285 CASF-2016 complexes and zero of PDBbind v2019 — so `prepare.py` converts the pdb with Open Babel on the way in.

### Build

```bash
podman build --format=docker -f Containerfile.torch -t sign:latest .     # cpu, tests only
podman build --format=docker -f Containerfile.gpu   -t sign-gpu:latest . # what training needs
```

### Run

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

### What comes out

| file | holds |
|---|---|
| `predictions.csv` | `complex_id,y_pred` |
| `embeddings.npz` | `ids` and `vectors`, 128-dim, native — `forward` returns it |
| `model.pt` | training only; reloads under `weights_only=True` |
| `history.csv` | training only; `epoch,train_loss,train_r,val_loss,val_r` and SIGN's own columns |
| `summary.json` | training only; best epoch, and what the run started from |

### Before you trust the numbers

**No checkpoint ships with this fork.** `.gitignore` excludes `*.pt` on purpose, so a fresh clone has to train one — which is the honest position, because the authors publish no weights either. That is also why this is the only model here whose training data is known: the split is ours and it excludes the CASF-2016 core set explicitly.

The pocket conversion is worth suspicion rather than comfort: pdb to mol2 assigns bond orders by perception, and Open Babel warns *\"Failed to kekulize aromatic bonds\"* while doing it. First place to look if a retrained model underperforms for no other visible reason.

<!-- gnn-benchmark:end -->

---

## SIGN-Paddle
Source code for KDD 2021 paper: "Structure-aware Interactive Graph Neural Networks for the Prediction of Protein-Ligand Binding Affinity".
<p align="center">
  <img src="sign.png" width="1000">
  <br />
</p> 

### Dependencies
- python >= 3.8
- paddlepaddle >= 2.1.0
- pgl >= 2.1.4
- openbabel == 3.1.1 (optional, only for preprocessing)

### Datasets
The PDBbind dataset can be downloaded [here](http://pdbbind-cn.org).

The CSAR-HiQ dataset can be downloaded [here](http://www.csardock.org).

You may need to use the [UCSF Chimera tool](https://www.cgl.ucsf.edu/chimera/) to convert the PDB-format files into MOL2-format files for feature extraction at first.

Alternatively, we also provided a [dropbox link](https://www.dropbox.com/sh/2uih3c6fq37qfli/AAD-LHXSWMLAuGWzcQLk5WI3a) for downloading PDBbind and CSAR-HiQ datasets.

The downloaded dataset should be preprocessed to obtain features and spatial coordinates:
```
python preprocess_pdbbind.py --data_path_core YOUR_DATASET_PATH --data_path_refined YOUR_DATASET_PATH --dataset_name pdbbind2016 --output_path YOUR_OUTPUT_PATH --cutoff 5
```
The parameter cutoff is the threshold of cutoff distance between atoms.

You can also use the processed data from [this link](https://www.dropbox.com/sh/68vc7j5cvqo4p39/AAB_96TpzJWXw6N0zxHdsppEa). Before training the model, please put the downloaded files into the directory (./data/).

### How to run
To train the model, you can run this command:
```
python train.py --cuda YOUR_DEVICE --model_dir MODEL_PATH_TO_SAVE --dataset pdbbind2016 --cut_dist 5 --num_angle 6
```
### Container (Podman/Docker)

The repo includes a `Containerfile` that packages the full environment — Python 3.9, openbabel 3.1.1 (via conda-forge), paddlepaddle-gpu 2.5.2 (CUDA 11.8), and PGL — into a single reproducible image using rootless Podman.

**Build:**
```bash
podman build -t sign .
```

If your GPU's CUDA version differs from 11.8, edit the `cu118` index URL in `Containerfile` before building (check your version with `nvidia-smi`). Supported suffixes: `cu117`, `cu118`, `cu120`.

**One-time host setup for GPU passthrough** (only needed once per machine):
```bash
nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

**Run an interactive session:**
```bash
mkdir -p outputs

podman run --rm -it \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --network=none \
  -v "$(pwd)/outputs:/work/outputs" \
  sign bash
```

Inside the container, train as usual:
```bash
python train.py --cuda 0 --model_dir /work/outputs --dataset pdbbind2016 --cut_dist 5 --num_angle 6
```

Saved checkpoints will appear in `./outputs/` on the host.

**How the image is structured:**
- Base: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` for CUDA support
- `micromamba` installs `openbabel` from conda-forge (no pip wheel exists for it)
- `paddlepaddle-gpu` and `pgl` are installed via pip from PaddlePaddle's package index
- All source code and preprocessed data are baked into the image at `/work`
- `--network=none` at runtime prevents outbound connections (pickle-loaded weights are a supply-chain risk)

### Citation
If you find our work is helpful in your research, please consider citing our paper:
```bibtex
@inproceedings{li2021structure,
  title={Structure-aware Interactive Graph Neural Networks for the Prediction of Protein-Ligand Binding Affinity},
  author={Li, Shuangli and Zhou, Jingbo and Xu, Tong and Huang, Liang and Wang, Fan and Xiong, Haoyi and Huang, Weili and Dou, Dejing and Xiong, Hui},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={975--985},
  year={2021}
}
```