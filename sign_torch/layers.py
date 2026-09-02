"""SIGN's layers, ported from PaddlePaddle/PGL to PyTorch and PyG.

A port rather than a reimplementation: the shapes, the order of operations and the parameter
names follow `layers.py` in the original so the two can be compared line by line. Where the
original is odd, this is odd in the same way, and the oddity is commented rather than fixed —
a port that silently improves the model is not a port.

The PGL constructs map as follows:

    g.send(fn, src_feat, dst_feat, edge_feat)   gather along edge_index and compute per-edge
    msg.reduce_softmax(alpha)                   softmax over messages sharing a destination
                                                -> torch_geometric.utils.softmax
    msg.reduce(x, pool_type="sum")              -> scatter add over destinations
    pgl.nn.GraphPool(pool_type='sum')           -> global_add_pool
    pgl.math.segment_pool(..., 'sum')           -> scatter add over segment ids

SIGN keeps three graph types and they are not interchangeable:

    a2a   atoms to atoms; its *edges* are what the model calls bonds
    b2a   bonds to atoms, so src indexes bonds and dst indexes atoms
    b2b   bonds to bonds, one graph per angle domain

**b2a has exactly one edge per bond.** The distance embedding is computed once per a2a edge
and then used as the *edge* feature of the b2a graph, so the two must line up: bond i sends
to its own destination atom and nowhere else. Getting this wrong shows up immediately as a
shape mismatch in Bond2AtomLayer rather than as a quiet wrong answer, which is the one
convenient thing about it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as segment_softmax


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum `src` rows into `dim_size` buckets given by `index`."""
    shape = (dim_size,) + src.shape[1:]
    out = src.new_zeros(shape)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=F.relu, bias=True):
        super().__init__()
        self.activation = activation
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.activation(self.fc(x))


class SpatialInputLayer(nn.Module):
    """Spatial Relation Embedding: distances bucketed to integers, then embedded."""

    def __init__(self, hidden_dim, cut_dist, activation=F.relu):
        super().__init__()
        self.cut_dist = cut_dist
        self.dist_embedding_layer = nn.Embedding(int(cut_dist) - 1, hidden_dim)
        self.dist_input_layer = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)

    def forward(self, dist_feat):
        dist = torch.clamp(dist_feat.squeeze(), 1.0, self.cut_dist - 1e-6).long() - 1
        return self.dist_input_layer(self.dist_embedding_layer(dist))


class Atom2BondLayer(nn.Module):
    """Node -> Edge aggregation: each bond reads both its atoms and its own feature."""

    def __init__(self, atom_dim, bond_dim, activation=F.relu):
        super().__init__()
        self.fc_agg = DenseLayer(atom_dim * 2 + bond_dim, bond_dim, activation, bias=True)

    def forward(self, edge_index, atom_feat, edge_feat):
        src, dst = edge_index
        h_agg = torch.cat([atom_feat[src], atom_feat[dst], edge_feat], dim=-1)
        return self.fc_agg(h_agg)


class Bond2AtomLayer(nn.Module):
    """Distance-aware Edge -> Node aggregation, multi-head attention over incident bonds."""

    def __init__(self, bond_dim, atom_dim, hidden_dim, num_heads, dropout,
                 merge="mean", activation=F.relu):
        super().__init__()
        self.merge = merge
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.src_fc = nn.Linear(bond_dim, num_heads * hidden_dim)
        self.dst_fc = nn.Linear(atom_dim, num_heads * hidden_dim)
        self.edg_fc = nn.Linear(hidden_dim, num_heads * hidden_dim)
        self.weight_src = nn.Parameter(torch.empty(num_heads, hidden_dim))
        self.weight_dst = nn.Parameter(torch.empty(num_heads, hidden_dim))
        self.weight_edg = nn.Parameter(torch.empty(num_heads, hidden_dim))
        for w in (self.weight_src, self.weight_dst, self.weight_edg):
            nn.init.xavier_uniform_(w)

        self.feat_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation

    def forward(self, edge_index, num_atoms, atom_feat, bond_feat, edge_feat):
        """`edge_index` maps bonds (src) to atoms (dst)."""
        bond_feat = self.feat_drop(bond_feat)
        atom_feat = self.feat_drop(atom_feat)
        edge_feat = self.feat_drop(edge_feat)

        bond_feat = self.src_fc(bond_feat).view(-1, self.num_heads, self.hidden_dim)
        atom_feat = self.dst_fc(atom_feat).view(-1, self.num_heads, self.hidden_dim)
        edge_feat = self.edg_fc(edge_feat).view(-1, self.num_heads, self.hidden_dim)

        attn_src = (bond_feat * self.weight_src).sum(-1)
        attn_dst = (atom_feat * self.weight_dst).sum(-1)
        attn_edg = (edge_feat * self.weight_edg).sum(-1)

        src, dst = edge_index
        alpha = self.leaky_relu(attn_src[src] + attn_dst[dst] + attn_edg)
        alpha = segment_softmax(alpha, dst, num_nodes=num_atoms)
        alpha = self.attn_drop(alpha).unsqueeze(-1)

        feature = bond_feat[src] * alpha
        if self.merge == "cat":
            feature = feature.reshape(-1, self.num_heads * self.hidden_dim)
        elif self.merge == "mean":
            feature = feature.mean(dim=1)

        rst = _scatter_sum(feature, dst, num_atoms)
        return self.activation(rst) if self.activation else rst


class DomainAttentionLayer(nn.Module):
    """Angle-domain attention among bonds."""

    def __init__(self, bond_dim, hidden_dim, dropout, activation=F.relu):
        super().__init__()
        self.attn_fc = nn.Linear(2 * bond_dim, hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, 1, bias=False)
        self.feat_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.activation = activation

    def forward(self, edge_index, num_bonds, bond_feat):
        bond_feat = self.feat_drop(bond_feat)
        src, dst = edge_index

        # The original concatenates the *source* with itself rather than with the
        # destination (layers.py:160). With attn_fc sized 2 * bond_dim that reads like a
        # typo for [src, dst], but reproducing it is the point of a port: changing it here
        # would silently make this a different model from the published one.
        h_src = bond_feat[src]
        h_c = self.tanh(self.attn_fc(torch.cat([h_src, h_src], dim=-1)))
        alpha = self.attn_out(h_c)

        alpha = segment_softmax(alpha, dst, num_nodes=num_bonds)
        alpha = self.attn_drop(alpha)
        rst = _scatter_sum(h_src * alpha, dst, num_bonds)
        return self.activation(rst) if self.activation else rst


class Bond2BondLayer(nn.Module):
    """One DomainAttention per angle domain, then merged."""

    def __init__(self, bond_dim, hidden_dim, num_angle, dropout, merge="cat", activation=None):
        super().__init__()
        self.num_angle = num_angle
        self.hidden_dim = hidden_dim
        self.merge = merge
        self.conv_layer = nn.ModuleList(
            DomainAttentionLayer(bond_dim, hidden_dim, dropout, activation=None)
            for _ in range(num_angle)
        )
        self.activation = activation

    def forward(self, edge_index_list, num_bonds, bond_feat):
        h_list = [self.conv_layer[k](edge_index_list[k], num_bonds, bond_feat)
                  for k in range(self.num_angle)]

        if self.merge == "cat":
            feat_h = torch.cat(h_list, dim=-1)
        elif self.merge == "mean":
            feat_h = torch.stack(h_list, dim=-1).mean(dim=1)
        elif self.merge == "sum":
            feat_h = torch.stack(h_list, dim=-1).sum(dim=1)
        elif self.merge == "max":
            feat_h = torch.stack(h_list, dim=-1).max(dim=1).values
        elif self.merge == "cat_max":
            stacked = torch.stack(h_list, dim=-1)
            feat_max = stacked.max(dim=1).values.reshape(-1, 1, self.hidden_dim)
            feat_h = (stacked * feat_max).reshape(-1, self.num_angle * self.hidden_dim)
        else:
            raise ValueError(f"unknown merge {self.merge!r}")

        return self.activation(feat_h) if self.activation else feat_h


class OutputLayer(nn.Module):
    """Sum-pool to a graph vector, then an MLP to a scalar.

    This pools *before* the MLP, which is why SIGN has a native complex embedding: the output
    of `self.pool` is a genuine graph-level representation and everything after it is the head.
    """

    def __init__(self, atom_dim, hidden_dim_list):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dim_list:
            layers.append(DenseLayer(atom_dim, hidden_dim, activation=F.relu))
            atom_dim = hidden_dim
        self.mlp = nn.ModuleList(layers)
        self.output_layer = nn.Linear(atom_dim, 1)

    def forward(self, batch_index, num_graphs, atom_feat):
        graph_feat = _scatter_sum(atom_feat, batch_index, num_graphs)
        for layer in self.mlp:
            graph_feat = layer(graph_feat)
        return self.output_layer(graph_feat), graph_feat


class PiPoolLayer(nn.Module):
    """Pairwise Interactive Pooling: a per-graph distribution over 36 bond types.

    This is the auxiliary head. SIGN is trained on affinity *and* on reproducing an
    interaction matrix, and this layer produces the latter — a softmax over the 4x9 bond
    types present in each complex, with absent types masked out rather than left at zero.

    The original builds it with `paddle.masked_select` plus segment pooling and pads each
    type's slice to the batch size; the same result comes out of one scatter here. The
    `-1e9` before the softmax is theirs and is what makes absent types vanish.
    """

    def __init__(self, bond_dim, hidden_dim, num_angle):
        super().__init__()
        self.bond_dim = bond_dim
        self.num_angle = num_angle
        self.num_type = 4 * 9
        self.fc_1 = DenseLayer(num_angle * bond_dim, hidden_dim, activation=F.relu, bias=True)
        self.fc_2 = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, bond_types_batch, type_count_batch, bond_feat):
        """
        bond_types_batch: (num_bonds,) the type index of every bond in the batch
        type_count_batch: (num_type, batch_size) how many bonds of each type per graph
        """
        batch_size = type_count_batch.shape[1]
        bond_feat = self.fc_1(bond_feat.reshape(-1, self.num_angle * self.bond_dim))

        # which graph each bond belongs to, derived from the per-graph counts exactly as
        # their get_index_from_counts + segment ids do
        inter_mat = bond_feat.new_full((batch_size, self.num_type), -1e9)
        for type_i in range(self.num_type):
            counts = type_count_batch[type_i]
            if counts.sum() == 0:
                continue
            mask = bond_types_batch == type_i
            if not bool(mask.any()):
                continue
            graph_ids = torch.repeat_interleave(
                torch.arange(batch_size, device=counts.device), counts
            )
            pooled = _scatter_sum(bond_feat[mask], graph_ids, batch_size)
            scores = self.fc_2(pooled).squeeze(-1)
            inter_mat[:, type_i] = torch.where(
                counts > 0, scores, torch.full_like(scores, -1e9)
            )

        return F.softmax(inter_mat, dim=1)
