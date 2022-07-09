"""
This is an implementation of the ComENet model

ComENet: Towards Complete and Efficient Message Passing for 3D Molecular Graphs: https://arxiv.org/abs/2206.08515

"""

from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing, GraphNorm
from torch_geometric.nn.acts import swish
from torch_geometric.nn import inits

from torch_scatter import scatter, scatter_min
from torch_sparse import matmul

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad, get_pbc_distances, radius_graph_pbc
from ocpmodels.models.comenet.utils import AngleEmbedding, TorsionEmbedding

from torch.nn import Embedding

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import math
from math import sqrt

try:
    import sympy as sym
except ImportError:
    sym = None


class Linear(torch.nn.Module):
    """
        A linear method encapsulation similar to PyG's

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
        bias (int)
        weight_initializer (string): (glorot or zeros)
    """

    def __init__(self, in_channels, out_channels, bias=True, weight_initializer='glorot'):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_initializer == 'glorot':
            inits.glorot(self.weight)
        elif self.weight_initializer == 'zeros':
            inits.zeros(self.weight)
        if self.bias is not None:
            inits.zeros(self.bias)

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)


class NonLinear(torch.nn.Module):
    """
        A Nonlinear layer with two linear modules

        Parameters
        ----------
        in_channels (int)
        middle_channels (int)
        out_channels (int)
        bias (bool)
        act (bool)
    """

    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False
    ):
        super(NonLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


class EmbeddingBlock(torch.nn.Module):
    """
        Atom embedding block

        Parameters
        ----------
        hidden_channels (int)
        act (function)
    """

    def __init__(self, hidden_channels, act=swish):
        super(EmbeddingBlock, self).__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x = self.act(self.emb(x))
        return x


class EdgeGraphConv(MessagePassing):
    """
        Graph convolution similar to PyG's GraphConv(https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)

        The difference is that this module performs Hadamard product between node feature and edge feature

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
    """
    def __init__(self, in_channels, out_channels):
        super(EdgeGraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_weight, size=None):
        x = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)
        return out + self.lin_r(x[1])

    def message(self, x_j, edge_weight):
        return edge_weight * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x[0], reduce=self.aggr)


class InteractionBlock(torch.nn.Module):
    """
        Atom interaction block.

        Parameters
        ----------
        hidden_channels (int)
        num_radial (int)
        num_spherical (int)
        num_layers (int)
        output_channels (int)
        act (function)
    """

    def __init__(
            self,
            hidden_channels,
            num_radial,
            num_spherical,
            num_layers,
            output_channels,
            act=swish,
    ):
        super(InteractionBlock, self).__init__()
        self.act = act

        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.lin_cat = Linear(2 * hidden_channels, hidden_channels)

        self.norm = GraphNorm(hidden_channels)

        # Transformations of Bessel and spherical basis representations.
        self.lin_feature1 = NonLinear(num_radial * num_spherical ** 2, hidden_channels, hidden_channels)
        self.lin_feature2 = NonLinear(num_radial * num_spherical, hidden_channels, hidden_channels)

        # Dense transformations of input messages.
        self.lin = Linear(hidden_channels, hidden_channels)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.norm.reset_parameters()

        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.lin_cat.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()

        self.final.reset_parameters()

    def forward(self, x, feature1, feature2, edge_index, batch):
        x = self.act(self.lin(x))

        feature1 = self.lin_feature1(feature1)
        h1 = self.conv1(x, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)

        feature2 = self.lin_feature2(feature2)
        h2 = self.conv2(x, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)

        h = self.lin_cat(torch.cat([h1, h2], 1))

        h = h + x
        for lin in self.lins:
            h = self.act(lin(h)) + h
        h = self.norm(h, batch)
        h = self.final(h)
        return h


@registry.register_model("comenet")
class ComENet(nn.Module):

    def __init__(
            self,
            num_atoms,
            bond_feat_dim,  # not used
            num_targets=1,
            otf_graph=False,
            use_pbc=True,
            regress_forces=False,
            hidden_channels=128,
            num_blocks=4,
            num_radial=32,
            num_spherical=7,
            cutoff=6.0,
            num_output_layers=3,
    ):
        super(ComENet, self).__init__()
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.num_blocks = num_blocks

        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        act = swish
        self.act = act

        self.feature1 = TorsionEmbedding(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature2 = AngleEmbedding(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        self.emb = EmbeddingBlock(hidden_channels, act)

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels,
                    num_radial,
                    num_spherical,
                    num_output_layers,
                    hidden_channels,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin_out = Linear(hidden_channels, num_targets, weight_initializer='zeros')
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        batch = data.batch
        z = data.atomic_numbers.long()
        num_nodes = data.atomic_numbers.size(0)

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50
            )

            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc:
            out = get_pbc_distances(
                data.pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
                return_distance_vec=True
            )

            edge_index = out["edge_index"]
            j, i = edge_index
            dist = out["distances"]
            vecs = out["distance_vec"]
        else:
            edge_index = radius_graph(data.pos, r=self.cutoff, batch=batch)
            j, i = edge_index
            vecs = data.pos[j] - data.pos[i]
            dist = vecs.norm(dim=-1)

        # Embedding block.
        x = self.emb(z)

        # Get reference nodes.
        # --------------------------------------------------------
        # Nearest neighbor f_i and second nearest neighbor s_i for i

        _, argmin0_i = scatter_min(dist, i, dim_size=num_nodes)
        argmin0_i[argmin0_i >= len(i)] = 0
        f_i = j[argmin0_i][i]

        add = torch.zeros_like(dist)
        add[argmin0_i] = self.cutoff
        dist1 = dist + add

        _, argmin1_i = scatter_min(dist1, i, dim_size=num_nodes)
        argmin1_i[argmin1_i >= len(i)] = 0
        s_i = j[argmin1_i][i]

        # --------------------------------------------------------
        # Nearest neighbor f_j and second nearest neighbor s_j for j

        _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(j)] = 0
        f_j = i[argmin0_j][j]

        add_j = torch.zeros_like(dist)
        add_j[argmin0_j] = self.cutoff
        dist1_j = dist + add_j

        _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(j)] = 0
        s_j = i[argmin1_j][j]

        # ----------------------------------------------------------
        # Reference nodes to compute the angle tau
        # tau: (iref, i, j, jref)
        # if f_i = j, we choose iref = s_i, otherwise, iref = f_i
        # if f_j = i, we choose jref = s_j, otherwise, jref = s_j
        mask_iref = f_i == j
        iref = torch.clone(f_i)
        iref[mask_iref] = s_i[mask_iref]
        idx_iref = argmin0_i[i]
        idx_iref[mask_iref] = argmin1_i[i][mask_iref]

        mask_jref = f_j == i
        jref = torch.clone(f_j)
        jref[mask_jref] = s_j[mask_jref]
        idx_jref = argmin0_j[j]
        idx_jref[mask_jref] = argmin1_j[j][mask_jref]

        pos_i_j, pos_i_fi, pos_i_si, pos_i_iref, pos_jref_j = (
            vecs,
            vecs[argmin0_i][i],
            vecs[argmin1_i][i],
            vecs[idx_iref],
            vecs[idx_jref]
        )

        # Calculate angle theta with nodes f_i, i, and j.
        a = ((pos_i_j) * pos_i_fi).sum(dim=-1)
        b = torch.cross(pos_i_j, pos_i_fi).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + math.pi

        # Calculate angle phi with nodes f_i, s_i, i, and j.
        plane1 = torch.cross(pos_i_si, pos_i_fi)
        plane2 = torch.cross(pos_i_j, pos_i_fi)
        a = (plane1 * plane2).sum(dim=-1)
        b = torch.cross(plane1, plane2).norm(dim=-1)
        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + math.pi

        # Calculate angle tau with nodes iref, i, j, and jref.
        plane1 = torch.cross(-pos_i_j, pos_i_iref)
        plane2 = torch.cross(pos_i_j, pos_jref_j)
        a = (plane1 * plane2).sum(dim=-1)
        b = torch.cross(plane1, plane2).norm(dim=-1)
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + math.pi

        feature1 = self.feature1(dist, theta, phi)
        feature2 = self.feature2(dist, tau)

        # Interaction blocks.
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, feature1, feature2, edge_index, batch)

        for lin in self.lins:
            x = self.act(lin(x))
        P = self.lin_out(x)

        energy = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)
        return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
