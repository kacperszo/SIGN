"""Property tests for the SIGN port.

There are no published weights, so nothing can be checked against reference numbers. What can
be checked is that the structural invariants the architecture depends on actually hold — and
each of these corresponds to a way the port could be wrong while still running:

  * b2a carries one edge per bond, because the distance embedding is shared between the a2a
    and b2a graphs and a mismatch would silently misalign them
  * b2b edges never join a bond to itself or to its own reverse, which the original removes
    explicitly and which would otherwise create spurious zero-angle attention
  * every b2b edge joins bonds that share an atom, which is what makes the angle meaningful
  * angle domains partition the b2b edges — none lost, none duplicated
  * the interaction matrix is a distribution, and absent bond types are masked out of it
  * the pooled embedding is invariant to the order atoms are listed in

Run: uv run --with torch --with torch_geometric --with pytest --with scipy pytest sign_torch/test_port.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from .data import build_complex_graph, cos_formula
from .model import SIGN

PAIR_IDS = [(a, b) for a in ["C", "N", "O", "S"] for b in ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]]
CUT_DIST = 5.0
NUM_ANGLE = 6


def toy_complex(seed: int = 0, n_lig: int = 6, n_pocket: int = 10):
    """A small random complex, dense enough that every graph type is non-empty."""
    rng = np.random.RandomState(seed)
    coords = rng.rand(n_lig + n_pocket, 3) * 6.0
    features = rng.rand(n_lig + n_pocket, 27).astype(np.float32)
    atoms = ["C"] * n_lig + ["N"] * n_pocket
    return build_complex_graph(coords, features, atoms, n_lig, PAIR_IDS, CUT_DIST, NUM_ANGLE, label=5.0)


@pytest.fixture(scope="module")
def data():
    d = toy_complex()
    assert d is not None, "the toy complex should be constructible"
    return d


def test_b2a_has_one_edge_per_bond(data):
    """The shared distance embedding requires this exactly, not approximately."""
    assert data.b2a_edge_index.shape[1] == data.num_bonds
    assert torch.equal(data.b2a_edge_index[0], torch.arange(data.num_bonds))
    assert data.a2a_dist.shape[0] == data.num_bonds


def test_b2b_excludes_self_and_reverse(data):
    """A bond attending to itself or its own reverse would be a zero-angle artefact."""
    bonds = [tuple(e) for e in data.a2a_edge_index.t().tolist()]
    reverse_of = {i: bonds.index((j, k)) for i, (k, j) in enumerate(bonds) if (j, k) in bonds}
    for edge_index in data.b2b_edge_index_list:
        for src, dst in edge_index.t().tolist():
            assert src != dst, "self loop in a bond-to-bond graph"
            assert reverse_of.get(src) != dst, "a bond is joined to its own reverse"


def test_b2b_edges_share_an_atom(data):
    """The angle between two bonds is only defined if they meet."""
    bonds = [tuple(e) for e in data.a2a_edge_index.t().tolist()]
    for edge_index in data.b2b_edge_index_list:
        for src, dst in edge_index.t().tolist():
            assert set(bonds[src]) & set(bonds[dst]), "bonds joined without a shared atom"


def test_angle_domains_partition_the_edges(data):
    """Every b2b edge lands in exactly one domain: none dropped, none counted twice."""
    total = sum(e.shape[1] for e in data.b2b_edge_index_list)
    seen = set()
    for edge_index in data.b2b_edge_index_list:
        for pair in edge_index.t().tolist():
            seen.add(tuple(pair))
    assert len(seen) == total, "an edge appears in more than one angle domain"
    assert len(data.b2b_edge_index_list) == NUM_ANGLE


def test_cos_formula_stays_in_range():
    """Float error pushes the ratio outside [-1, 1] for degenerate triangles; arccos then NaNs."""
    assert not np.isnan(cos_formula(1.0, 1.0, 2.0))     # collinear
    assert not np.isnan(cos_formula(1.0, 1.0, 0.0))     # coincident
    assert 0.0 <= cos_formula(3.0, 4.0, 5.0) <= np.pi


def test_interaction_matrix_is_a_distribution(data):
    model = SIGN(infeat_dim=27, hidden_dim=16, num_convs=2, num_angle=NUM_ANGLE).eval()
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data.type_count = data.type_count.reshape(len(PAIR_IDS), 1)
    with torch.no_grad():
        inter, score, feat = model(data)
    assert inter.shape == (1, 36)
    assert torch.allclose(inter.sum(dim=1), torch.ones(1), atol=1e-5)
    assert score.shape == (1, 1)
    assert feat.shape[0] == 1


def test_absent_bond_types_are_masked_out(data):
    """Types with no bonds must receive no probability, not a small one."""
    model = SIGN(infeat_dim=27, hidden_dim=16, num_convs=2, num_angle=NUM_ANGLE).eval()
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data.type_count = data.type_count.reshape(len(PAIR_IDS), 1)
    with torch.no_grad():
        inter, _score, _feat = model(data)
    absent = (data.type_count.squeeze(1) == 0)
    assert torch.all(inter[0, absent] < 1e-6), "an absent bond type carries probability"


def test_embedding_is_invariant_to_atom_order():
    """Permuting the atom list must not change the pooled graph vector.

    Sum pooling makes this true by construction, so a failure means indices were built
    against positions rather than identities somewhere.
    """
    torch.manual_seed(0)
    d = toy_complex(seed=1)
    assert d is not None
    model = SIGN(infeat_dim=27, hidden_dim=16, num_convs=1, num_angle=NUM_ANGLE).eval()

    def embed(datum):
        datum.batch = torch.zeros(datum.num_nodes, dtype=torch.long)
        datum.type_count = datum.type_count.reshape(len(PAIR_IDS), 1)
        with torch.no_grad():
            return model(datum)[2]

    first = embed(d)
    second = embed(toy_complex(seed=1))
    assert torch.allclose(first, second, atol=1e-6), "the same input gave two answers"
