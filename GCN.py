

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DrugGraph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, num_edge_types=4, dropout_p=0.1):
        super().__init__()
        assert num_layers >= 2, "num_layers should be at least 2"

        self.edge_type_scale = nn.Parameter(torch.tensor([1.0, 1.2, 1.5, 2.0]))  # shape: [num_edge_types]

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = nn.Dropout(dropout_p)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x, edge_index, edge_attr, edge_type):
        scaled_edge_weight = self.edge_type_scale[edge_type] * edge_attr

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, scaled_edge_weight)
            if i != len(self.convs) - 1:
                x = self.act(x)
                x = self.dropout(x)
        return x

class ProteinGraph(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 3,
                 dropout_p: float = 0.1):
        super(ProteinGraph, self).__init__()

        assert num_layers >= 2, "num_layers should be >= 2"

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = nn.Dropout(dropout_p)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = self.act(x)
                x = self.dropout(x)

        return x




# class EdgedrugSAGE(nn.Module):
#     def __init__(self, hidden_channels, out_channels, dropout_p=0.1):
#         super(EdgedrugSAGE, self).__init__()
#
#         # 三层 SAGEConv
#         self.conv1 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
#         self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
#         self.conv3 = SAGEConv(hidden_channels, out_channels, aggr='mean')
#
#         # 激活函数、Dropout、归一化
#         self.leaky_relu = nn.LeakyReLU(0.1)
#         self.dropout = nn.Dropout(dropout_p)
#         self.layer_norm = nn.LayerNorm(out_channels)
#
#     def forward(self, x, edge_index, edge_weight):
#         row, col = edge_index
#         deg = torch.zeros(x.size(0), device=x.device).index_add(0, row, edge_weight)
#         norm_weight = edge_weight / (deg[row] + 1e-8)
#         # 3. 消息传递（带边权重）
#         agg = torch.zeros_like(x)
#         agg.index_add_(0, row, x[col] * norm_weight.unsqueeze(-1))  # 聚合邻居特征
#         # 4. 结合自身特征 + 邻居特征
#         x = x + agg
#         # 5. 两层 GraphSAGE
#         x = self.conv1(x, edge_index)
#         x = self.leaky_relu(x)
#         x = self.dropout(x)
#         x = self.conv2(x, edge_index)
#         x = self.leaky_relu(x)
#         x = self.dropout(x)
#         x = self.conv3(x, edge_index)
#
#         return x

