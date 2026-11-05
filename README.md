# SIGN — structure-aware interaction graph network, rewritten in PyTorch

> A maintained fork. The authors' own README is kept as
> [README.upstream.md](README.upstream.md) for attribution and for their description of
> the method — **its build and run instructions are not current for this fork.**

## What it is

Atoms and bonds update in alternation, with bond-to-bond attention split across
angular domains, over three graph types: atom-to-atom at a 5 A cutoff whose *edges* are what the model
calls bonds, bond-to-atom with exactly one edge per bond, and bond-to-bond grouped by the angle
between them. Two outputs trained jointly — an affinity scalar and an auxiliary interaction matrix
over 36 ligand/pocket atom-pair types.

The original is PaddlePaddle + PGL and ships no checkpoints, so this is a rewrite rather than a port.
`preprocess_pdbbind.py` and `featurizer.py` are the authors' own and unchanged, so the featurisation
is theirs even though the graphs are not.

## Reported performance

| | |
|---|---|
| CASF-2016 scoring | **R 0.759**, RMSE 1.420, n=285 — paper 0.797 +/- 0.012 |
| CASF-2016 ranking | rho 0.596 |
| embedding | 128d, **native** (`forward` returns it) |
| training | 4285 complexes, **300 epochs** as the authors train, 4 h on an RTX 4060 Ti; best on validation at epoch 163 |

These numbers come from this fork's own retrain, not from authors' weights, which do not exist.
The training split excludes the CASF-2016 core set explicitly. The paper is the KDD 2021 paper
cited in `README.upstream.md`, which also has the authors' citation; the upstream repository is
`agave233/SIGN`.

## Weights

No checkpoint ships with this fork: `*.pt` is gitignored, and the authors publish no weights
either. Train one with the `train` command below, which writes `model.pt` (plus `history.csv`
and `summary.json`) into the output directory. The `predict`/`embed` command expects a
directory holding that file, passed as `CKPT_DIR` (mounted at `/ckpt`).

## Build

There is no unsuffixed `Containerfile`, so pass `-f`. (The authors' PaddlePaddle environment was
removed; the original is in the upstream history.)

```bash
podman build --format=docker -f Containerfile.torch -t sign:latest .     # cpu, tests only
podman build --format=docker -f Containerfile.gpu   -t sign-gpu:latest . # what training needs
```

## Run it

Every command below runs with `--network=none` and a read-only root filesystem.

Input is one directory per complex:

    <complexes>/<id>/<id>_pocket.pdb
    <complexes>/<id>/<id>_ligand.mol2

**Nothing ships `<id>_pocket.mol2`** — zero of 285 CASF-2016 complexes and zero of PDBbind v2019 — so `prepare.py` converts the pdb with Open Babel on the way in.

```bash
# predict and embed: one forward pass writes predictions.csv and embeddings.npz
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$CKPT_DIR:/ckpt:ro" \
    localhost/sign:latest \
    sh -c 'cd /work && TMPDIR=/tmp python -m sign_torch.predict --complexes /data --model /ckpt/model.pt --out /outputs --device cpu'

# train
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/sign:latest \
    sh -c 'cd /work && if [ -f /cache/sign_train_cut5_ang6_t12.pt ]; then echo reusing /cache/sign_train_cut5_ang6_t12.pt; else TMPDIR=/tmp python -m sign_torch.prepare --complexes /data --labels /splits/train.csv --out /cache/sign_train_cut5_ang6_t12.pt --cut_dist 5 --num_angle 6; fi && if [ -f /cache/sign_val_cut5_ang6_t12.pt ]; then echo reusing /cache/sign_val_cut5_ang6_t12.pt; else TMPDIR=/tmp python -m sign_torch.prepare --complexes /data --labels /splits/val.csv --out /cache/sign_val_cut5_ang6_t12.pt --cut_dist 5 --num_angle 6; fi && python -m sign_torch.train --graphs /cache/sign_train_cut5_ang6_t12.pt --out /outputs --seed 0 --val_graphs /cache/sign_val_cut5_ang6_t12.pt --epochs 30 --device cpu'

# finetune  (encoder frozen; drop --freeze-encoder to tune all of it)
# ENC_DIR holds encoder.pt: a state dict with the head tensors removed (see Porting notes)
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$ENC_DIR:/ckpt:ro" \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/sign:latest \
    sh -c 'cd /work && if [ -f /cache/sign_train_cut5_ang6_t12.pt ]; then echo reusing /cache/sign_train_cut5_ang6_t12.pt; else TMPDIR=/tmp python -m sign_torch.prepare --complexes /data --labels /splits/train.csv --out /cache/sign_train_cut5_ang6_t12.pt --cut_dist 5 --num_angle 6; fi && if [ -f /cache/sign_val_cut5_ang6_t12.pt ]; then echo reusing /cache/sign_val_cut5_ang6_t12.pt; else TMPDIR=/tmp python -m sign_torch.prepare --complexes /data --labels /splits/val.csv --out /cache/sign_val_cut5_ang6_t12.pt --cut_dist 5 --num_angle 6; fi && python -m sign_torch.train --graphs /cache/sign_train_cut5_ang6_t12.pt --out /outputs --seed 0 --val_graphs /cache/sign_val_cut5_ang6_t12.pt --epochs 30 --init-encoder /ckpt/encoder.pt --freeze-encoder --device cpu'
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

The pocket conversion is worth suspicion rather than comfort: pdb to mol2 assigns bond orders by perception, and Open Babel warns *"Failed to kekulize aromatic bonds"* while doing it. First place to look if a retrained model underperforms for no other visible reason.

## Porting notes

**It is a rewrite, and the featurisation is still the authors'.** The original is PaddlePaddle
+ PGL and ships no checkpoints, so there was nothing to verify a faithful port against and
training was required either way. `preprocess_pdbbind.py` and `featurizer.py` contain no Paddle
and are reused unchanged, as are the other upstream `.py` files; only `dataset.py`, which built
the PGL graphs, was rewritten as `sign_torch/data.py`. The PGL constructs map as
`send`/`reduce_softmax`/`reduce(sum)`/`GraphPool` to gather-along-`edge_index`,
`torch_geometric.utils.softmax`, and scatter-add. `data/pdbbind2016_test.pkl` is upstream's test
split.

Carried over deliberately:

- **`DomainAttentionLayer` concatenates the source with itself**, not with the destination
  (`layers.py:160`, `paddle.concat([src_feat['h'], src_feat['h']])`). With `attn_fc` sized
  `2 * bond_dim` that reads like a typo for `[src, dst]`; it is reproduced as-is, because fixing
  it would quietly make this a different model from the published one.
- **The authors' code is internally inconsistent about the diagonal.** The distance embedding is
  computed per a2a edge and reused as the b2a edge feature, so three counts have to agree.
  `dist_feat` includes the zero diagonal (`dist_mat[i,i] = 0 < cut_dist`); a2a edges built via
  `coo_matrix` do not (coo drops zeros); the bond list `indices` does. Self-pairs are excluded
  here, which makes all three agree and keeps a zero distance out of the angle calculation, where
  the authors' own guard treats it as unusable and abandons the complex.

Traps:

- **`add_fea=3`, not 2.** `process_dataset` passes 3 (`preprocess_pdbbind.py:324`) and it sets
  the feature width, which an assertion downstream checks. Guessing 2 (the function default)
  produced an `AssertionError` with no message on every complex.
- **`_pocket.mol2` has to be generated**, by Open Babel from the pdb. Open Babel warns *"Failed to
  kekulize aromatic bonds"* while doing so, so pocket bond orders and atom types are perceived,
  not read. First place to look if a retrained model underperforms for no visible reason.
- **Parallel workers need `--shm-size=4g`.** They return `Data` objects and torch passes tensors
  through `/dev/shm`, which podman caps at 64 MB; the run then dies partway through with "No space
  left on device", which looks like a data problem. Serial preparation does not need it but is
  roughly ten times slower (8 minutes versus 3 hours over 4285 complexes).
- **No `torch_scatter`.** The port uses torch's own `scatter_add_` and takes only `softmax` from
  PyG, removing the one dependency that compiles from source when its wheel index is unreachable.
- **b2a has exactly one edge per bond**, because the distance embedding is shared between the a2a
  and b2a graphs. A mismatch surfaces as a shape error rather than a wrong answer.
- **`COPY . /work`** in the Containerfiles copies whatever `.containerignore` does not exclude;
  `*.pt` is excluded so an image never carries weights.

**The embedding is native, and the head is smaller than it looks.** `OutputLayer` pools before
its MLP, so the graph vector exists in the model's own logic. **The head is
`output_layer.output_layer`, not `output_layer`:** `OutputLayer.forward` pools, runs the
512/256/128 MLP, and returns *that MLP's output* as `graph_feat`, so the 128-dim embedding is
produced inside the module whose name suggests it is the head, and only the final
`Linear(128, 1)` turns it into an affinity. Declaring the whole module a head would leave that MLP
randomly initialised in every fine-tuned model, and nothing raises: the split succeeds, training
runs, the curve is merely worse. `pipool_layer` is a head too: it predicts the auxiliary
interaction matrix, a training target. Split: 67 encoder tensors (1,726,208 parameters) and 5
head tensors (98,689). To make an `encoder.pt`, take a checkpoint's state dict and drop every key
for which `sign_torch.transfer.is_head` is true.

**A frozen encoder is put in eval mode**, not just left with `requires_grad=False`.
`Bond2BondLayer` and `Bond2AtomLayer` carry dropout, and leaving them in training mode means the
head learns against a representation resampled every step. `transfer.py:set_training_mode` handles it.

**Tests.** `sign_torch/test_port.py` has eight property tests, run as the last step of the image
build. With no published weights there are no reference numbers, so they check invariants the
architecture rests on: b2a's one-edge-per-bond shape; b2b never joining a bond to itself or its
reverse; b2b edges always sharing an atom; angle domains partitioning the edges exactly; the
interaction matrix summing to one with absent types masked to zero; and the embedding being
reproducible. `sign_torch/test_transfer.py` has nine tests on the encoder/head boundary; the
load-bearing one, `test_embedding_survives_transfer`, asserts that after transfer the new model
embeds identically to the old one and scores differently, because only the head was replaced.

**Boundary check by refitting.** Freezing the encoder and training the head alone on 400
complexes gave CASF-2016 R 0.7247 against the full model's 0.7249, so the encoder carries
essentially all the signal and the head is cheap to replace.

Two hyperparameters were left at the authors' defaults and never swept: `lambda_` 1.75 on the
auxiliary loss, and `dec_step` 8000.
