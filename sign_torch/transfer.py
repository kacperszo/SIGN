"""Move a trained encoder into a fresh SIGN, and freeze it if asked.

The harness cuts checkpoints on the host (`gnnb encoder split`); this is the other half, and
it has to live in the model's own container because loading weights back needs the model class.

**Where SIGN's boundary actually is.** `OutputLayer.forward` pools the atom features, runs them
through an MLP, and returns *both* the scalar and the MLP's output:

    graph_feat = scatter_sum(atom_feat, batch)      # 512-dim, no parameters
    for layer in self.mlp: graph_feat = layer(...)  # 512 -> 256 -> 128
    return self.output_layer(graph_feat), graph_feat

so the 128-dim vector `predict.py` writes as the embedding is produced by `output_layer.mlp`,
and only the final `output_layer.output_layer` Linear(128, 1) turns it into an affinity. The
head is therefore that Linear, not the whole `OutputLayer` — declaring the module would have
put the embedding MLP on the head side and transferred an encoder whose representation is
randomly initialised. Everything downstream still runs, and the curve merely looks
disappointing.

`pipool_layer` is a head too: it predicts the auxiliary interaction matrix, which is a
training target rather than a representation. A fine-tune re-learns it in a few epochs, and
carrying it over would transfer an opinion about bond-type distributions along with the
encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: The task heads, by parameter-name prefix. Must agree with `encoder.head` for sign.torch in
#: harness/registry.toml — the harness cuts the file, this loads it, and a disagreement between
#: the two produces tensors that belong to neither side.
HEAD_PREFIXES = ("output_layer.output_layer", "pipool_layer")


def is_head(name: str) -> bool:
    """Whether a parameter belongs to a task head. Matched at module boundaries, so
    `output_layer` never claims `output_layer_norm` — and so the nested prefix works."""
    return any(name == p or name.startswith(p + ".") for p in HEAD_PREFIXES)


def read_encoder(path: str) -> dict[str, torch.Tensor]:
    """Load an encoder file, accepting either of the two shapes it arrives in.

    `gnnb encoder split` writes `{"encoder": {...}, "encoder_params": n, ...}`; a bare state
    dict turns up when someone points this at a full checkpoint instead. Both are read, and
    anything that is not a tensor is dropped rather than being loaded into the model.

    `weights_only=True` is not negotiable: an encoder file is tensors, and opening a foreign
    checkpoint any other way is unprotected unpickling.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("encoder"), dict):
        payload = payload["encoder"]
    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a state dict, got {type(payload).__name__}")
    return {k: v for k, v in payload.items() if torch.is_tensor(v)}


def transfer_encoder(model: nn.Module, path: str, freeze: bool = False) -> dict[str, object]:
    """Overlay an encoder onto a fresh model, leaving the head at its initialisation.

    Overlaid and loaded **strictly**, not `load_state_dict(..., strict=False)`: a partial load
    that silently moved nothing is indistinguishable from success until the training curve
    disappoints, weeks later. Every way this can be wrong raises here instead.

    Returns what happened, for `summary.json` — a transfer nobody recorded is a result nobody
    can reproduce.
    """
    encoder = read_encoder(path)
    target = model.state_dict()

    head_keys = [k for k in encoder if is_head(k)]
    if head_keys:
        raise SystemExit(
            f"{path} carries {len(head_keys)} head tensors ({head_keys[:3]}). That is a whole "
            f"checkpoint, not an encoder — cut it with `gnnb encoder split` first, or the "
            f"fine-tune starts from the old task's head."
        )
    unknown = [k for k in encoder if k not in target]
    if unknown:
        raise SystemExit(f"{path}: {len(unknown)} tensors have no slot in this model: {unknown[:5]}")
    mismatched = [k for k, v in encoder.items() if target[k].shape != v.shape]
    if mismatched:
        raise SystemExit(
            f"{path}: shape mismatch on {len(mismatched)} tensors: "
            + ", ".join(f"{k} {tuple(encoder[k].shape)} into {tuple(target[k].shape)}"
                        for k in mismatched[:3])
            + ". The encoder was trained at a different width; rebuild it or match the sizes."
        )
    uncovered = [k for k in target if k not in encoder and not is_head(k)]
    if uncovered:
        raise SystemExit(
            f"{path}: {len(uncovered)} model tensors are neither in the encoder nor in the "
            f"head — the boundary is wrong: {uncovered[:5]}"
        )

    model.load_state_dict({**target, **encoder}, strict=True)
    moved = sum(v.numel() for v in encoder.values())
    print(f"transferred {len(encoder)} tensors / {moved:,} params from {path}")

    frozen = 0
    if freeze:
        for name, param in model.named_parameters():
            if not is_head(name):
                param.requires_grad_(False)
                frozen += param.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"froze {frozen:,} encoder params; {trainable:,} trainable in the heads")

    return {"init_encoder": path, "transferred_tensors": len(encoder),
            "transferred_params": moved, "frozen_params": frozen}


def set_training_mode(model: nn.Module, training: bool, frozen_encoder: bool) -> None:
    """Put the model in the right mode, keeping a frozen encoder deterministic.

    A frozen encoder still has dropout in `Bond2BondLayer` and `Bond2AtomLayer`, and leaving it
    in train mode means the head learns against a representation that is re-sampled every step.
    That is a defensible regulariser and a different experiment: `gnnb probe` fits a ridge on
    the deterministic embedding, and `--freeze-encoder` is meant to be the trainable analogue of
    it. So the encoder goes to eval and only the heads train.
    """
    if not training or not frozen_encoder:
        model.train(training)
        return
    model.eval()
    for name, module in model.named_modules():
        if name and is_head(name) and any(p.requires_grad for p in module.parameters()):
            module.train()
