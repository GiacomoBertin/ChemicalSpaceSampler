import torch_geometric.nn as pygnn
import torch_geometric as pyg
import torch.nn as nn
from torch_geometric.data import Data
import net_blocks as nb
from typing import List, Callable, Union
import torch
from einops import rearrange, repeat


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

        self.output_module = nn.Sequential(nn.BatchNorm1d(hidden_channels),
                                           pygnn.models.MLP([hidden_channels] +
                                                            [max(hidden_channels // 2 ** i, output_channels)
                                                             for i in range(num_final_layers)] +
                                                            [output_channels],
                                                            batch_norm=False),
                                           nn.Linear(output_channels, output_channels))

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


class FullyConnected(nn.Module):
    def __init__(self, use_batch_norm, input_ch, output_ch):
        super(FullyConnected, self).__init__()
        self.net = nb.fc(use_batch_norm, input_ch, output_ch)

    def forward(self, x):
        return self.net(x)


class Conv1D(nn.Module):
    def __init__(self, use_batch_norm, input_ch, output_ch):
        super(Conv1D, self).__init__()
        self.net = nb.conv1d(use_batch_norm, input_ch, output_ch)

    def forward(self, x):
        return self.net(x)


class Pool1D(nn.Module):
    def __init__(self, pool_shape: int = 1, use_flatten: bool = True):
        super(Pool1D, self).__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool1d(pool_shape), nn.Flatten(1) if use_flatten else nn.Identity())

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self,
                 input_ch: int,
                 latent_ch: int,
                 hidden_ch: Union[int, List[int]] = None,
                 layer_instance: str = 'fc',
                 intermediate_layer: nn.Module = None,
                 use_batch_norm: bool = True,
                 num_layers: int = 5,
                 num_layers_head: int = 5,
                 net_init: Callable = nb.net_init,
                 use_vae: bool = False,
                 activation: str = None):
        super(Encoder, self).__init__()

        if hidden_ch is None:
            hidden_ch = int((input_ch + latent_ch) / 2)

        layer_instance = getattr(nb, layer_instance)

        if isinstance(net_init, str):
            net_init = getattr(nb, net_init)

        if activation is not None:
            activation = getattr(nn, activation)()

        self.use_vae = use_vae
        if isinstance(hidden_ch, int):
            self.net = [layer_instance(use_batch_norm,
                                       input_ch if i == 0 else hidden_ch,
                                       hidden_ch,
                                       activation=activation)
                        for i in range(num_layers)]
        else:
            self.net = [layer_instance(use_batch_norm,
                                       input_ch,
                                       hidden_ch[0],
                                       activation=activation)]

            self.net += [layer_instance(use_batch_norm,
                                        hidden_ch[i],
                                        hidden_ch[i + 1],
                                        activation=activation)
                         for i in range(len(hidden_ch) - 1)]

        self.net = nn.Sequential(*self.net)
        # net_init(self.net)

        self.head = []

        if intermediate_layer is not None:
            self.head += [intermediate_layer]

        if isinstance(hidden_ch, int):
            channels_head = list(torch.floor(torch.linspace(hidden_ch, latent_ch, num_layers_head)))
        else:
            channels_head = list(torch.floor(torch.linspace(hidden_ch[-1], latent_ch, num_layers_head)))

        self.head += [nb.fc(use_batch_norm,
                            int(channels_head[i].item()),
                            int(channels_head[i + 1].item()),
                            activation=nn.Identity() if i == (num_layers_head - 2) else activation)
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
                 hidden_ch: Union[int, List[int]] = None,
                 layer_instance: str = 'fc',
                 use_batch_norm: bool = True,
                 rearrange_str: str = 'b c -> b c',
                 rearrange_kwarg=None,
                 num_layers: int = 5,
                 net_init: Callable = nb.net_init,
                 final_activation: str = None,
                 activation: str = None):
        super(Decoder, self).__init__()

        if rearrange_kwarg is None:
            rearrange_kwarg = {}

        if hidden_ch is None:
            hidden_ch = int((input_ch + latent_ch) / 2)

        if activation is not None:
            activation = getattr(nn, activation)()

        layer_instance = getattr(nb, layer_instance)

        if isinstance(net_init, str):
            net_init = getattr(nb, net_init)

        self.rearrange = rearrange_str
        self.rearrange_kwarg = rearrange_kwarg
        self.final_activation = final_activation

        if isinstance(hidden_ch, int):
            self.net = [layer_instance(use_batch_norm,
                                       latent_ch if i == 0 else hidden_ch,
                                       input_ch if i == (num_layers - 1) else hidden_ch,
                                       activation=activation)
                        for i in range(num_layers)]

        else:
            self.net = [layer_instance(use_batch_norm,
                                       latent_ch,
                                       hidden_ch[0],
                                       activation=activation)]

            self.net += [layer_instance(use_batch_norm,
                                        hidden_ch[i],
                                        hidden_ch[i + 1],
                                        activation=activation)
                         for i in range(len(hidden_ch) - 1)]

            self.net += [layer_instance(use_batch_norm,
                                        hidden_ch[-1],
                                        input_ch,
                                        activation=activation)]

        if self.final_activation is None:
            self.net.append(nn.Identity())

        elif self.final_activation == 'sigmoid':
            self.net.append(layer_instance(False,
                                           input_ch,
                                           input_ch,
                                           activation=nn.Identity()))
            self.net.append(nn.Sigmoid())

        elif self.final_activation == 'tanh':
            self.net.append(layer_instance(False,
                                           input_ch,
                                           input_ch,
                                           activation=nn.Identity()))
            self.net.append(nn.Tanh())

        elif 'softmax' in self.final_activation:
            self.net.append(nn.Softmax(dim=int(self.final_activation.split('|')[-1])))

        self.net = nn.Sequential(*self.net)
        # net_init(self.net)

    def forward(self, x):
        h = repeat(x, self.rearrange, **self.rearrange_kwarg)
        return self.net(h)

