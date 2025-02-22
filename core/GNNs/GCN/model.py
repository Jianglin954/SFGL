import torch
import torch.nn.functional as F


from torch_geometric.nn import GCNConv

import ipdb

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, use_pred):
        super(GCN, self).__init__()
        self.use_pred = use_pred
        if self.use_pred:
            self.encoder = torch.nn.Embedding(out_channels+1, hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=False, cached=True))       # add_self_loops=False,
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, add_self_loops=False, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, add_self_loops=False, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, weight=None):
        if self.use_pred:
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1)
        # ipdb.set_trace()
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, edge_weight=weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        # return x.log_softmax(dim=-1)
        return x
