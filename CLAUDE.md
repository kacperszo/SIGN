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

## It is a rewrite, and the featurisation is still theirs

The original is PaddlePaddle + PGL and ships **no checkpoints**, so there was nothing to verify a
faithful port against and training was required either way. `preprocess_pdbbind.py` and
`featurizer.py` contain no Paddle and are reused unchanged; only `dataset.py`, which built the PGL
graphs, was rewritten. The PGL constructs map as `send`/`reduce_softmax`/`reduce(sum)`/`GraphPool`
to gather-along-edge_index, `torch_geometric.utils.softmax`, and scatter-add.

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

## The embedding is native, and the head is smaller than it looks

`OutputLayer` pools before its MLP, so the graph vector exists in the model's own logic and
nothing has to be invented — the same situation as IGN. It retains 98% of the head's Pearson R,
the highest here, which is unsurprising for a pooling trained jointly with the head rather than
recovered afterwards.

**The head is `output_layer.output_layer`, not `output_layer`.** `OutputLayer.forward` pools, runs
the 512/256/128 MLP, and returns *that MLP's output* as `graph_feat` — so the 128-dim embedding is
produced inside the module whose name suggests it is the head, and only the final `Linear(128, 1)`
turns it into an affinity. Declaring the whole module would leave that MLP randomly initialised in
every fine-tuned model, and nothing raises: the split succeeds, training runs, the curve is worse.
`pipool_layer` is a head too — it predicts the auxiliary interaction matrix, a training target.

Split: **67 encoder tensors / 1,726,208 params (95%), 5 head tensors / 98,689**.

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

`sign_torch/test_transfer.py` — nine tests on the encoder/head boundary, run at build time
beside them. Same idea, different failure mode: everything here still trains and still prints a
curve, just a worse one. The load-bearing case is **`test_embedding_survives_transfer`** — after
transfer the new model must embed identically to the old one and score differently, because only
the head was replaced. It fails on the pre-2026-09-09 boundary and passes on the corrected one,
which is what makes that correction a measurement rather than an argument.

## Training and fine-tuning through the harness

`sign.torch` declares `train` and `finetune`. The adapter chains graph preparation and training
in one container command, and `--cache` keeps the prepared graphs between runs — eight minutes
a run otherwise, which every transfer experiment would pay again.

```bash
gnnb encoder split --variant sign.torch --out /tmp/sign_encoder.pt

gnnb train --variant sign.torch --dataset data/pdbbind_v2019 \
    --train-split benchmarks/pdbbind_train.csv --val-split benchmarks/pdbbind_val.csv \
    --epochs 30 --workers 10 --cache ~/.cache/gnnb --gpu

gnnb train --variant sign.torch --dataset data/pdbbind_v2019 \
    --train-split benchmarks/pdbbind_train.csv --val-split benchmarks/pdbbind_val.csv \
    --init-encoder /tmp/sign_encoder.pt --freeze-encoder --gpu

gnnb run --variant sign.torch --capability predict --dataset data/CASF-2016/coreset \
    --checkpoint runs/<stamp>_sign.torch_finetune/outputs/model.pt --gpu
```

**A frozen encoder is put in eval mode**, not just left with `requires_grad=False`.
`Bond2BondLayer` and `Bond2AtomLayer` carry dropout, and leaving them training means the head
learns against a representation resampled every step — a defensible regulariser, but a
different experiment from the one `gnnb probe` measures. `transfer.py:set_training_mode` keeps
the two comparable.

**The boundary is confirmed by refitting.** Freeze the encoder, train the head alone on 400
complexes, and CASF comes back at R 0.7247 against the full model's 0.7249 — so the encoder carries
essentially all the signal and the head is cheap to replace. A boundary cutting through the encoder
could not do that.

## Build & run

```bash
podman build --format=docker -f Containerfile.gpu -t sign-gpu:latest .

# graphs (needs shared memory for the workers)
podman run --rm --network=none --shm-size=4g \
  -v <pdbbind>:/data:ro -v <labels>:/labels:ro -v <out>:/outputs:rw,U \
  localhost/sign-gpu:latest sh -c "cd /work && TMPDIR=/tmp python -m sign_torch.prepare \
    --complexes /data --out /outputs/train.pt --labels /labels/pdbbind_train.csv --workers 10"

# train — `--out` is a directory now: model.pt, history.csv and summary.json go in it.
# A path ending in .pt is still accepted and the other two land beside it.
podman run --rm --network=none --shm-size=4g --device nvidia.com/gpu=all \
  -v <out>:/outputs:rw,U localhost/sign-gpu:latest \
  sh -c "cd /work && python -m sign_torch.train --graphs /outputs/train.pt \
    --val_graphs /outputs/val.pt --out /outputs --epochs 30"
```

`Containerfile.torch` is the CPU variant, for tests only — one epoch over 4285 complexes does
not finish in minutes on CPU.

## Next

1. The trained checkpoint is **not committed** — `.gitignore` excludes `*.pt`, deliberately.
   The registry bind-mounts it from the working tree, so a fresh clone has to train it or be
   handed the file. It is baked into the image by `COPY . /work` as a side effect, which is
   worth knowing before assuming an image is reproducible from the repository alone.
2. Two hyperparameters were left at the authors' defaults but never swept: `lambda_` 1.75 on the
   auxiliary loss, and `dec_step` 8000. Neither was tuned for our split size.
3. The `DomainAttentionLayer` self-concatenation, as an ablation.
4. A transfer result that means something needs an encoder trained on data the fine-tuning
   split does not contain. SIGN's own checkpoint saw all 4285 complexes, so it can only
   demonstrate the mechanism — the first real measurement wants an encoder from a different
   task, which is what the pretraining programme is for.
