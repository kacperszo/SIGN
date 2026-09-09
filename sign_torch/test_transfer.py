"""Tests for the encoder/head boundary: does transfer actually move the representation.

`test_port.py` asks whether the port computes what SIGN computes. These ask whether the model
can be taken apart and put back together, which is what the pretraining programme needs and is
a different way to be wrong — every failure here still runs, still trains, and still prints a
curve. The curve is just worse than it should be, for a reason nobody can see.

The load-bearing one is `test_embedding_survives_transfer`. SIGN's 128-dim embedding is
produced inside `OutputLayer`, by the MLP that sits one line before the affinity Linear. The
registry originally declared the whole of `output_layer` as the head, which would have left
that MLP randomly initialised in every fine-tuned model — so `gnnb probe` would have been
measuring a representation that transfer never moved. This test fails on that boundary and
passes on the corrected one.

Run: uv run --with torch --with torch_geometric --with pytest --with scipy \
        pytest sign_torch/test_transfer.py
"""

from __future__ import annotations

import pytest
import torch

from .model import SIGN
from .test_port import NUM_ANGLE, PAIR_IDS, toy_complex
from .transfer import HEAD_PREFIXES, is_head, read_encoder, set_training_mode, transfer_encoder


def build(seed: int) -> SIGN:
    torch.manual_seed(seed)
    return SIGN(infeat_dim=27, hidden_dim=16, num_convs=2, dense_dims=(32, 16, 8),
                num_angle=NUM_ANGLE)


def batch_of_one(seed: int = 1):
    data = toy_complex(seed=seed)
    assert data is not None
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data.type_count = data.type_count.reshape(len(PAIR_IDS), 1)
    return data


def encoder_of(model: SIGN) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if not is_head(k)}


@pytest.fixture()
def encoder_file(tmp_path):
    """An encoder cut from a trained-looking model, in the shape `gnnb encoder split` writes."""
    source = build(seed=0)
    path = tmp_path / "encoder.pt"
    torch.save({"encoder": encoder_of(source), "variant": "sign.torch"}, path)
    return source, str(path)


def test_boundary_covers_every_parameter():
    """Every tensor is on exactly one side, and neither side is empty.

    A prefix that matches nothing puts the whole model in the encoder, which transfers the old
    task's head along with the representation and looks like a suspiciously good result.
    """
    names = list(build(seed=0).state_dict())
    head = [n for n in names if is_head(n)]
    assert head, f"no parameter matched {HEAD_PREFIXES}"
    assert len(head) < len(names), "everything matched the head"
    # the affinity Linear and the auxiliary head, and nothing else
    assert {n.split(".weight")[0].split(".bias")[0] for n in head} == {
        "output_layer.output_layer", "pipool_layer.fc_1.fc", "pipool_layer.fc_2"}


def test_the_embedding_mlp_is_not_in_the_head():
    """`output_layer.mlp` produces the vector `predict.py` writes; it belongs to the encoder."""
    assert not is_head("output_layer.mlp.0.fc.weight")
    assert is_head("output_layer.output_layer.weight")


def test_transfer_moves_the_encoder_and_leaves_the_head(encoder_file):
    source, path = encoder_file
    fresh = build(seed=99)
    before = {k: v.clone() for k, v in fresh.state_dict().items()}

    transfer_encoder(fresh, path, freeze=False)
    after = fresh.state_dict()

    for name in encoder_of(source):
        assert torch.equal(after[name], source.state_dict()[name]), f"{name} did not transfer"
    for name in (n for n in after if is_head(n)):
        assert torch.equal(after[name], before[name]), f"{name} was overwritten; it is the head"


def test_embedding_survives_transfer(encoder_file):
    """The whole point: after transfer, the new model embeds identically to the old one.

    Only the affinity scalar is allowed to differ, because only the head was replaced. This is
    what makes a `probe` result predictive of what a fine-tune starts from.
    """
    source, path = encoder_file
    fresh = build(seed=99)
    transfer_encoder(fresh, path, freeze=False)

    data = batch_of_one()
    with torch.no_grad():
        source.eval(); fresh.eval()
        _, source_score, source_feat = source(data)
        _, fresh_score, fresh_feat = fresh(data)

    assert torch.allclose(source_feat, fresh_feat, atol=1e-6), \
        "the transferred model embeds differently — the boundary cuts through the encoder"
    assert not torch.allclose(source_score, fresh_score, atol=1e-6), \
        "the affinity is unchanged, so the head came across too"


def test_freezing_leaves_gradients_only_on_the_head(encoder_file):
    _source, path = encoder_file
    model = build(seed=99)
    transfer_encoder(model, path, freeze=True)

    assert all(not p.requires_grad for n, p in model.named_parameters() if not is_head(n))
    assert all(p.requires_grad for n, p in model.named_parameters() if is_head(n))

    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.1)
    _inter, score, _feat = model(batch_of_one())
    score.sum().backward()
    optimiser.step()

    after = model.state_dict()
    for name in (n for n in after if not is_head(n)):
        assert torch.equal(after[name], before[name]), f"{name} moved while frozen"
    assert not torch.equal(after["output_layer.output_layer.weight"],
                           before["output_layer.output_layer.weight"]), "the head did not learn"


def test_a_frozen_encoder_is_deterministic(encoder_file):
    """Dropout in the frozen half would resample the representation the head is learning on."""
    _source, path = encoder_file
    model = build(seed=99)
    transfer_encoder(model, path, freeze=True)
    set_training_mode(model, training=True, frozen_encoder=True)

    data = batch_of_one()
    first = model(data)[2]
    second = model(data)[2]
    assert torch.allclose(first, second, atol=1e-6), "the frozen encoder is still sampling"
    assert model.output_layer.output_layer.training, "the head should be training"


def test_a_full_checkpoint_is_refused(tmp_path):
    """Handing this a whole checkpoint would silently carry the old task's head across."""
    path = tmp_path / "full.pt"
    torch.save(build(seed=0).state_dict(), path)
    with pytest.raises(SystemExit, match="not an encoder"):
        transfer_encoder(build(seed=1), str(path), freeze=False)


def test_a_mismatched_width_is_refused(tmp_path, encoder_file):
    """An encoder trained at another hidden size must not load into half a model."""
    _source, path = encoder_file
    wider = SIGN(infeat_dim=27, hidden_dim=32, num_convs=2, dense_dims=(32, 16, 8),
                 num_angle=NUM_ANGLE)
    with pytest.raises(SystemExit, match="shape mismatch"):
        transfer_encoder(wider, path, freeze=False)


def test_read_encoder_accepts_both_shapes(tmp_path, encoder_file):
    source, path = encoder_file
    wrapped = read_encoder(path)
    bare_path = tmp_path / "bare.pt"
    torch.save(encoder_of(source), bare_path)
    assert set(wrapped) == set(read_encoder(str(bare_path)))
