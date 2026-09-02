"""SIGN, ported to PyTorch.

Structure-aware Interaction Graph Network: atoms and bonds are updated in alternation, with
bond-to-bond attention split across angle domains. Two outputs — an affinity scalar and an
auxiliary interaction matrix — which is why the original trains on both.

Ported from the PaddlePaddle original in this repository. **There are no published weights**,
so nothing here can be checked against reference numbers: the model has to be trained either
way, and training it in Paddle would have cost a second framework, a second package index and
a second GPU stack for one model. See CLAUDE.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (Atom2BondLayer, Bond2AtomLayer, Bond2BondLayer, OutputLayer,
                     PiPoolLayer, SpatialInputLayer)


class SIGN(nn.Module):
    def __init__(self, infeat_dim, hidden_dim, num_convs=2, dense_dims=(128, 128, 64),
                 cut_dist=5.0, num_angle=6, merge_b2b="cat", merge_b2a="mean",
                 num_heads=4, feat_drop=0.2):
        super().__init__()
        self.num_convs = num_convs
        self.num_angle = num_angle

        self.input_layer = SpatialInputLayer(hidden_dim, cut_dist, activation=F.relu)
        self.atom2bond_layers = nn.ModuleList()
        self.bond2bond_layers = nn.ModuleList()
        self.bond2atom_layers = nn.ModuleList()

        for i in range(num_convs):
            atom_dim = infeat_dim if i == 0 else (
                hidden_dim * num_heads if "cat" in merge_b2a else hidden_dim)
            bond_dim = hidden_dim * num_angle if "cat" in merge_b2b else hidden_dim

            self.atom2bond_layers.append(
                Atom2BondLayer(atom_dim, bond_dim=hidden_dim, activation=F.relu))
            self.bond2bond_layers.append(
                Bond2BondLayer(hidden_dim, hidden_dim, num_angle, feat_drop,
                               merge=merge_b2b, activation=None))
            self.bond2atom_layers.append(
                Bond2AtomLayer(bond_dim, atom_dim, hidden_dim, num_heads, feat_drop,
                               merge=merge_b2a, activation=F.relu))

        self.pipool_layer = PiPoolLayer(hidden_dim, hidden_dim, num_angle)
        self.output_layer = OutputLayer(hidden_dim, list(dense_dims))

    def forward(self, data):
        """`data` carries the three graph types SIGN needs; see layers.py for what each is."""
        atom_h = data.x.float()
        dist_h = self.input_layer(data.a2a_dist.float())

        for i in range(self.num_convs):
            bond_h = self.atom2bond_layers[i](data.a2a_edge_index, atom_h, dist_h)
            bond_h = self.bond2bond_layers[i](
                data.b2b_edge_index_list, bond_h.shape[0], bond_h)
            atom_h = self.bond2atom_layers[i](
                data.b2a_edge_index, atom_h.shape[0], atom_h, bond_h, dist_h)

        pred_inter_mat = self.pipool_layer(data.bond_types, data.type_count, bond_h)
        pred_score, graph_feat = self.output_layer(
            data.batch, int(data.batch.max()) + 1, atom_h)
        return pred_inter_mat, pred_score, graph_feat
