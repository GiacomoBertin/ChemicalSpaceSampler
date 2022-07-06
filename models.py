import torch_geometric.nn as pygnn
import torch_geometric as pyg
import torch.nn as nn
from torch_geometric.data import Data


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




