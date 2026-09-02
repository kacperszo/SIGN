"""Build SIGN's three graph types as a PyG Data object.

Ported from `dataset.py`, which is the only preprocessing file that touched PaddlePaddle —
`preprocess_pdbbind.py` and `featurizer.py` are plain numpy/openbabel and are reused unchanged.

The construction, in the original's own order:

    a2a   an atom-to-atom graph over every pair closer than `cut_dist`. Its *edges* are what
          the model calls bonds, and the edge feature is the raw distance.
    b2a   one edge per bond, from the bond to its destination atom. That one-to-one shape is
          not incidental: the distance embedding is computed per a2a edge and reused as the
          b2a edge feature, so the two must line up.
    b2b   bonds that share an atom, split into `num_angle` graphs by the angle between them.
          Reverse pairs (i,j)/(j,i) and self-loops are removed first.

Bond types index the 4x9 ligand-atom/pocket-atom pairs and drive the auxiliary interaction
matrix; -1 marks an intra-molecular bond, which belongs to no type.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import Data


def cos_formula(a: float, b: float, c: float) -> float:
    """Angle opposite side c, by the law of cosines. Clipped: float error escapes [-1, 1]."""
    res = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    return np.arccos(np.clip(res, -1.0, 1.0))


def build_complex_graph(
    coords: np.ndarray,
    features: np.ndarray,
    atoms: list,
    num_atoms_ligand: int,
    pair_ids: list,
    cut_dist: float,
    num_angle: int,
    label: float | None = None,
) -> Data | None:
    """One complex -> one Data. Returns None when a degenerate angle makes it unusable."""
    num_atoms = len(coords)
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    # --- a2a ---
    # Self-pairs are excluded here, and that is a deliberate divergence. The original takes
    # `dist_feat` from every pair below the cutoff — the zero diagonal included — but builds
    # the edge list with `coo_matrix`, which drops zeros and so excludes the diagonal, while
    # the bond list includes it again. The model needs all three counts equal: the distance
    # embedding is computed per a2a edge and reused as the b2a edge feature. Excluding
    # self-pairs is the only way they agree, and it also avoids feeding a zero distance into
    # the angle calculation, which the original's own guard treats as unusable.
    off_diagonal = ~np.eye(num_atoms, dtype=bool)
    within_cutoff = (dist_mat < cut_dist) & off_diagonal
    dist_feat = dist_mat[within_cutoff].reshape(-1, 1)
    dist_graph_base = np.where(within_cutoff, dist_mat, 0.0)
    atom_graph = coo_matrix(dist_graph_base)
    a2a_edge_index = np.vstack([atom_graph.row, atom_graph.col])

    # --- bond nodes, and which ligand/pocket atom pair each one spans ---
    indices, bond_pair_atom_types = [], []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j or dist_mat[i, j] >= cut_dist:
                continue
            at_i, at_j = atoms[i], atoms[j]
            if i < num_atoms_ligand <= j and (at_j, at_i) in pair_ids:
                bond_pair_atom_types.append(pair_ids.index((at_j, at_i)))
            elif j < num_atoms_ligand <= i and (at_i, at_j) in pair_ids:
                bond_pair_atom_types.append(pair_ids.index((at_i, at_j)))
            else:
                bond_pair_atom_types.append(-1)
            indices.append([i, j])

    num_bonds = len(indices)
    if num_bonds == 0:
        return None

    # --- b2a: bond i -> its destination atom ---
    indices_arr = np.asarray(indices)
    b2a_edge_index = np.vstack([np.arange(num_bonds), indices_arr[:, 1]])

    # --- b2b: bonds sharing an atom, minus self and reverse pairs ---
    assignment_b2a = np.zeros((num_bonds, num_atoms), dtype=np.int64)
    assignment_a2b = np.zeros((num_atoms, num_bonds), dtype=np.int64)
    for i, (src, dst) in enumerate(indices):
        assignment_b2a[i, dst] = 1
        assignment_a2b[src, i] = 1

    bond_graph_base = assignment_b2a @ assignment_a2b
    np.fill_diagonal(bond_graph_base, 0)
    index_of = {tuple(v): i for i, v in enumerate(indices)}
    reverse = [index_of[(j, i)] for i, j in indices]
    bond_graph_base[range(num_bonds), reverse] = 0

    x, y = np.where(bond_graph_base > 0)
    angle_feat = np.zeros(len(x), dtype=np.float32)
    for k in range(len(x)):
        b1, b2 = indices[x[k]], indices[y[k]]
        a, b, c = dist_mat[b1[0], b1[1]], dist_mat[b2[0], b2[1]], dist_mat[b1[0], b2[1]]
        if a == 0 or b == 0:
            # the original bails out of the whole complex here rather than guessing
            return None
        angle_feat[k] = cos_formula(a, b, c)

    unit = 180.0 / num_angle
    angle_index = np.clip((np.rad2deg(angle_feat) / unit).astype(np.int64), 0, num_angle - 1)

    b2b_edge_index_list = []
    for domain in range(num_angle):
        sel = angle_index == domain
        edges = np.vstack([x[sel], y[sel]]) if sel.any() else np.zeros((2, 0), dtype=np.int64)
        b2b_edge_index_list.append(torch.as_tensor(edges, dtype=torch.long))

    # --- per-type bond counts, for the auxiliary interaction matrix ---
    bond_types = np.asarray(bond_pair_atom_types)
    type_count = np.zeros(len(pair_ids), dtype=np.int64)
    for t in bond_types:
        if t >= 0:
            type_count[t] += 1

    data = Data(
        x=torch.as_tensor(features, dtype=torch.float),
        a2a_edge_index=torch.as_tensor(a2a_edge_index, dtype=torch.long),
        a2a_dist=torch.as_tensor(dist_feat, dtype=torch.float),
        b2a_edge_index=torch.as_tensor(b2a_edge_index, dtype=torch.long),
        bond_types=torch.as_tensor(bond_types, dtype=torch.long),
        type_count=torch.as_tensor(type_count, dtype=torch.long).unsqueeze(1),
        num_nodes=num_atoms,
    )
    data.b2b_edge_index_list = b2b_edge_index_list
    data.num_bonds = num_bonds
    if label is not None:
        data.y = torch.tensor([float(label)])
    return data
