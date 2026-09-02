# CLAUDE.md — SIGN

## What this is

Structure-aware Interaction Graph Network: atoms and bonds updated in alternation, with
bond-to-bond attention split across angle domains. Two outputs — an affinity scalar and an
auxiliary interaction matrix over 36 ligand/pocket atom-pair types — and it trains on both.

Originally PaddlePaddle + PGL. **Rewritten here in PyTorch**; the Paddle path is abandoned.

## Current state

Done. Ported, trained and evaluated.

| | |
|---|---|
| CASF-2016 scoring | **R 0.725**, RMSE 1.511, c-index 0.764, n=285 |
| CASF-2016 ranking | **rho 0.653** — best of every model measured; baseline 0.619 |
| top-1 | 0.561 |
| embedding | 128d, **native**, probe R 0.711, retains **98%** of its own head |
| training | 4285 complexes, 30 epochs, 23 min on an RTX 4060 Ti |

**It is the only model in the benchmark whose training data we know.** The split excludes the
CASF-2016 core set explicitly, and that matters: 266 of the 285 core-set complexes sit inside
PDBbind refined, and no other repository in the roster publishes a training list. SIGN comes
last of the trained models on Pearson and first on ranking — how much of that gap is
architecture rather than memorisation is exactly what the others cannot answer.

Report it with that caveat attached. It is being judged more strictly than everything beside it.

## Why a rewrite was the right call here

The usual objection — that an unverified rewrite is an implementation rather than a
reproduction — does not bite when there is nothing to verify against. **SIGN ships no
checkpoints.** Training was required either way, so running the original would have bought a
second framework, a second package index and a second GPU stack for one model, and bought
nothing in fidelity.

`preprocess_pdbbind.py` and `featurizer.py` contain no Paddle at all and are reused unchanged,
so the featurisation is literally theirs. Only `dataset.py`, which built the PGL graphs, needed
rewriting. The PGL constructs map as:

| PGL | PyTorch |
|---|---|
| `g.send(fn, src_feat, dst_feat, edge_feat)` | gather along `edge_index`, compute per edge |
| `msg.reduce_softmax(alpha)` | `torch_geometric.utils.softmax` over destinations |
| `msg.reduce(x, pool_type="sum")` | scatter-add over destinations |
| `pgl.nn.GraphPool('sum')` | scatter-add over `batch` |

## Two things carried over deliberately

**`DomainAttentionLayer` concatenates the source with itself**, not with the destination
(`layers.py:160`). With `attn_fc` sized `2 * bond_dim` that reads like a typo for `[src, dst]`,
and it is reproduced as-is — fixing it would quietly make this a different model from the
published one. Worth revisiting as a deliberate ablation now that a baseline exists.

**Their own code is internally inconsistent about the diagonal.** Three counts have to be equal,
because the distance embedding is computed per a2a edge and reused as the b2a edge feature:

| quantity | includes the zero diagonal? |
|---|---|
| `dist_feat` | **yes** — `dist_mat[i,i] = 0 < cut_dist` |
| a2a edges via `coo_matrix` | **no** — coo drops zeros |
| the bond list `indices` | **yes** |

They cannot all be right. Self-pairs are excluded here, which makes all three agree and keeps a
zero distance out of the angle calculation — where their own guard treats it as unusable and
abandons the complex.

## The embedding is native

`OutputLayer` pools before its MLP, so the graph vector exists in the model's own logic and
nothing has to be invented — the same situation as IGN. It retains 98% of the head's Pearson R,
the highest here, which is unsurprising for a pooling trained jointly with the head rather than
recovered afterwards.

## Hard-won facts (do NOT regress these)

- **`add_fea=3`, not 2.** Their `process_dataset` passes 3 (`preprocess_pdbbind.py:324`) and it
  sets the feature width, which an assertion downstream checks. Guessing it produced an
  `AssertionError` with no message on every complex.
- **`_pocket.mol2` has to be generated.** Nothing ships one — zero of 285 CASF complexes, zero of
  PDBbind v2019. openbabel converts the pdb and warns *"Failed to kekulize aromatic bonds"* while
  doing so, so pocket bond orders and atom types are **perceived, not read**. First place to look
  if a retrained model underperforms for no visible reason.
- **Parallel workers need `--shm-size=4g`.** They return `Data` objects and torch passes tensors
  through /dev/shm, which podman caps at 64 MB — the run then dies *partway through* with
  "No space left on device", so the symptom looks like a data problem. Serial mode does not need
  it, and is ten times slower: 8 minutes versus 3 hours over 4285 complexes.
- **No `torch_scatter`.** The port uses torch's own `scatter_add_` and takes only `softmax` from
  PyG. That removes the one dependency that compiles from source when its wheel index is
  unreachable — which it was, once, mid-build.
- **b2a has exactly one edge per bond**, because the distance embedding is shared between the a2a
  and b2a graphs. A mismatch surfaces as a shape error rather than a wrong answer, which is the
  one convenient thing about it.

## Tests

`sign_torch/test_port.py` — eight property tests, run as the last step of the image build so the
port cannot rot silently. With no published weights there are no reference numbers, so these
check invariants the architecture rests on, each one a way the port could be wrong while still
running: b2a's one-edge-per-bond shape; b2b never joining a bond to itself or its reverse; b2b
edges always sharing an atom; angle domains partitioning the edges exactly; the interaction
matrix summing to one with absent types masked to zero; and the embedding being reproducible.

## Build & run

```bash
podman build --format=docker -f Containerfile.gpu -t sign-gpu:latest .

# graphs (needs shared memory for the workers)
podman run --rm --network=none --shm-size=4g \
  -v <pdbbind>:/data:ro -v <labels>:/labels:ro -v <out>:/outputs:rw,U \
  localhost/sign-gpu:latest sh -c "cd /work && TMPDIR=/tmp python -m sign_torch.prepare \
    --complexes /data --out /outputs/train.pt --labels /labels/pdbbind_train.csv --workers 10"

# train
podman run --rm --network=none --shm-size=4g --device nvidia.com/gpu=all \
  -v <out>:/outputs:rw,U localhost/sign-gpu:latest \
  sh -c "cd /work && python -m sign_torch.train --graphs /outputs/train.pt \
    --val_graphs /outputs/val.pt --out /outputs/sign_model.pt --epochs 30"
```

`Containerfile.torch` is the CPU variant, for tests only — one epoch over 4285 complexes does
not finish in minutes on CPU.

## Next

1. The trained checkpoint is **not committed** (7 MB, and reproducible from `train.py`). Decide
   where it should live if it is worth keeping.
2. Two hyperparameters were left at the authors' defaults but never swept: `lambda_` 1.75 on the
   auxiliary loss, and `dec_step` 8000. Neither was tuned for our split size.
3. The `DomainAttentionLayer` self-concatenation, as an ablation.
