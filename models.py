import torch_geometric.nn as pygnn
import torch_geometric as pyg
import torch.nn as nn
from torch_geometric.data import Data
import net_blocks as nb
from typing import List, Callable
import torch


class GATNet(nn.Module):
    def __init__(self,
                 input_channels,
                 edge_dim,
                 output_channels=1,
                 hidden_channels=None,
                 num_layers=3,
                 activation=nn.ELU,
                 use_batch_norm=False,
                 heads: int = 1,
                 concat: bool = False,
                 dropout: float = 0.6,
                 num_final_layers: int = 3
                 ):
        super(GATNet, self).__init__()
        if hidden_channels is None:
            hidden_channels = input_channels // 2

        self.input_module = pygnn.Sequential(
            'x, edge_index, edge_attr',
            [
                (pygnn.GATv2Conv(input_channels, hidden_channels, heads=heads, edge_dim=edge_dim, concat=concat), 'x, edge_index, edge_attr -> x'),
                nn.BatchNorm1d(hidden_channels * heads if concat else hidden_channels) if use_batch_norm else nn.Identity(),
                activation(),
                (nn.Dropout(p=dropout), 'x -> x'),
            ]
        )

        hidden_layers = []
        for i in range(num_layers):

            hidden_layers += [
                (pygnn.GATv2Conv(hidden_channels * heads if concat else hidden_channels,
                                 hidden_channels,
                                 heads=heads,
                                 edge_dim=edge_dim,
                                 concat=concat
                                 ), 'x, edge_index, edge_attr -> x'),
                nn.BatchNorm1d(hidden_channels * heads if concat else hidden_channels) if use_batch_norm else nn.Identity(),
                activation(),
                (nn.Dropout(p=dropout), 'x -> x'),
             ]

        self.hidden_layers = pygnn.Sequential('x, edge_index, edge_attr', hidden_layers)

        self.output_module = pygnn.models.MLP([hidden_channels] + [max(hidden_channels // 2 ** i, output_channels) for i in range(num_final_layers)] + [output_channels])

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h = self.input_module(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h = self.hidden_layers(x=h, edge_index=edge_index, edge_attr=edge_attr)
        h = pygnn.global_mean_pool(h, batch)
        h = self.output_module(h)

        return h


class GENNet(nn.Module):
    def __init__(self,
                 input_channels,
                 edge_dim,
                 output_channels=1,
                 hidden_channels=None,
                 num_layers=2,
                 num_final_layers: int = 3,
                 learn_t: bool = False,
                 ):
        super(GENNet, self).__init__()
        if hidden_channels is None:
            hidden_channels = input_channels // 2

        self.mlp_edges_attr = pygnn.MLP([edge_dim] + [hidden_channels] * num_layers)

        self.input_module = pygnn.Sequential(
            'x, edge_index, edge_attr',
            [
                (pygnn.MLP([edge_dim] + [input_channels] * num_layers), 'edge_attr -> edge_attr'),
                (pygnn.GENConv(input_channels,
                               hidden_channels,
                               aggr='softmax',
                               t=1.0,
                               learn_t=learn_t,
                               p=1.0,
                               learn_p=False,
                               num_layers=num_layers),
                 'x, edge_index, edge_attr -> x'),
            ]
        )

        hidden_layers = [(pygnn.MLP([input_channels] + [hidden_channels] * num_layers), 'edge_attr -> edge_attr')]
        for i in range(num_layers):

            hidden_layers += [
                (pygnn.GENConv(hidden_channels,
                               hidden_channels,
                               aggr='softmax',
                               t=1.0,
                               learn_t=learn_t,
                               p=1.0,
                               learn_p=False,
                               num_layers=num_layers),
                 'x, edge_index, edge_attr -> x'),
             ]

        self.hidden_layers = pygnn.Sequential('x, edge_index, edge_attr', hidden_layers)

        self.output_module = pygnn.models.MLP([hidden_channels] + [max(hidden_channels // 2 ** i, output_channels) for i in range(num_final_layers)] + [output_channels])

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h = self.input_module(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h = self.hidden_layers(x=h, edge_index=edge_index, edge_attr=edge_attr)
        h = pygnn.global_mean_pool(h, batch)
        h = self.output_module(h)

        return h


class Encoder(nn.Module):
    def __init__(self,
                 input_ch: int,
                 latent_ch: int,
                 hidden_ch: int = None,
                 layer_instance: Callable = nb.fc,
                 intermediate_layer: nn.Module = None,
                 use_batch_norm: bool = True,
                 num_layers: int = 5,
                 num_layers_head: int = 5,
                 net_init: Callable = nb.net_init,
                 use_vae: bool = False):
        super(Encoder, self).__init__()

        if hidden_ch is None:
            hidden_ch = max(input_ch, latent_ch)

        self.use_vae = use_vae
        self.net = [layer_instance(use_batch_norm,
                                   input_ch if i == 0 else hidden_ch,
                                   hidden_ch)
                    for i in range(num_layers)]

        self.net = nn.Sequential(*self.net)
        net_init(self.net)

        self.head = []

        if intermediate_layer is not None:
            self.head += [intermediate_layer]

        channels_head = list(torch.floor(torch.linspace(hidden_ch, latent_ch, num_layers_head)))
        self.head += [nb.fc(use_batch_norm,
                            int(channels_head[i].item()),
                            int(channels_head[i + 1].item()))
                      for i in range(num_layers_head - 1)
                      ]

        self.head = nn.Sequential(*self.head)

        self.mu = nn.Linear(latent_ch, latent_ch)
        self.logvar = nn.Linear(latent_ch, latent_ch)

    def forward(self, x):
        h = self.net(x)
        h = self.head(h)
        mu = self.mu(h)
        logvar = self.logvar(h)

        if self.use_vae:
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = 1e-2 * torch.randn_like(std)
                w = eps.mul(std).add_(mu)
                return w, mu, logvar
            else:
                return mu
        else:
            return mu


class Decoder(nn.Module):
    def __init__(self,
                 input_ch: int,
                 latent_ch: int,
                 hidden_ch: int = None,
                 layer_instance: Callable = nb.fc,
                 use_batch_norm: bool = True,
                 num_layers: int = 5,
                 net_init: Callable = nb.net_init,
                 final_layer: List[nn.Module] = None):
        super(Decoder, self).__init__()

        if hidden_ch is None:
            hidden_ch = max(input_ch, latent_ch)

        self.net = [layer_instance(use_batch_norm,
                                   input_ch if i == 0 else hidden_ch,
                                   hidden_ch)
                    for i in range(num_layers)]

        self.net = nn.Sequential(*self.net)
        net_init(self.net)

        if final_layer is not None:
            self.head += final_layer

        self.head = nn.Sequential(*self.head)

    def forward(self, x):
        return self.net(x)

