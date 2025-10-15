import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import math


class Bi_Inter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        inter_channels = channels // 2
        self.conv_c = nn.Sequential(
            nn.Conv1d(channels, inter_channels, 1),
            nn.BatchNorm1d(inter_channels),
            nn.GELU(),
            nn.Conv1d(inter_channels, channels, 1)
        )
        self.conv_s = nn.Conv1d(channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mode):
        if mode == 'channel':
            x_res = x.mean(-1, keepdim=True)  # [B, C, 1]
            x_res = self.sigmoid(self.conv_c(x_res))
        elif mode == 'spatial':
            x_res = self.sigmoid(self.conv_s(x))  # [B, 1, V]
        return x_res

class CGABlock(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super().__init__()
        mid_channels = in_channels // rel_reduction # 16

        self.conv1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, mid_channels, 1)
        self.conv3 = nn.Conv1d(in_channels, out_channels, 1)

        # CA-GCN, SA
        self.diff_conv = nn.Conv2d(mid_channels * 2, mid_channels, 1, groups=mid_channels)
        self.edge_proj = nn.Conv2d(mid_channels, out_channels, 1)
        self.att_proj = nn.Conv2d(mid_channels, out_channels, 1)

        self.bi_inter = Bi_Inter(out_channels)
        self.tanh = nn.Tanh()
        self.sigmod = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros(1))

        self.last_A_out = None  # 添加缓存变量
        self.last_att = None  # 添加缓存变量

        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x, A_static):
        # x: [B, C, V]
        B, C, V = x.shape
        x1, x2 = self.conv1(x), self.conv2(x)

        # channal-GCN
        x_diff = torch.cat([
            x1.unsqueeze(-1) - x2.unsqueeze(-2),  # [B, C, V, V]
            x2.unsqueeze(-1) - x1.unsqueeze(-2)  # [B, C, V, V]
        ], dim=1)  # 拼接得到 [B, 2C, V, V]



        A_dynamic = self.tanh(self.diff_conv(x_diff))
        A_out = self.edge_proj(A_dynamic)  # learnable CA'
        A_out = A_static.unsqueeze(0).unsqueeze(1) + self.alpha * A_out  # A + α·CA'

        # A_out = F.layer_norm(A_out, A_out.shape[-2:])  # 可视化投影
        # A_out = self.tanh(A_out)

        # 特征主干
        x3 = self.conv3(x)  # [B, C_out, V]

        # Bi-Inter
        att = self.tanh(torch.einsum('bcu,bcv->bcuv', x1, x2) /  math.sqrt(V))  # [B, C, V, V]
        att = self.att_proj(att)  # [B, C_out, V, V]

        # att = F.layer_norm(att, att.shape[-2:])# 可视化投影
        # att = self.tanh(att)

        x_att = torch.einsum('bcu,bcuv->bcv', x3, att)

        c_att = self.bi_inter(x_att, mode='channel')  # [B, C, 1]
        x_gcn = torch.einsum('bcuv,bcv->bcu', A_out, x3) * c_att  # [B, C, V]

        s_att = self.bi_inter(x_gcn, mode='spatial')  # [B, 1, V] SA
        x_att = x_att * s_att

        out = x_gcn + x_att
        out = out + self.residual(x)

        self.last_A_out = A_out.detach().cpu()  # shape: [C, V, V]
        self.last_att = att.detach().cpu()  # shape: [C, V, V]

        return out


class CSADGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, A_static, dropout=None):
        super(CSADGCN, self).__init__()
        self.edge_index = edge_index
        self.A_static = A_static
        self.dropout = dropout
        self.vis_weights = {
            'gcn_att': [],
            'sa_att': []
        }

        # GCN backbone for initial feature lifting
        self.gc1 = GCNConv(in_channels, hidden_channels)
        # self.gc2 = GCNConv(hidden_channels, hidden_channels * 2)

        self.bn = nn.BatchNorm1d(hidden_channels)

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Channel-wise Graph and Attention Blocks (structure: C -> 2C -> C)
        self.cga1 = CGABlock(hidden_channels, hidden_channels)
        self.cga2 = CGABlock(hidden_channels, hidden_channels * 2)
        self.cga3 = CGABlock(hidden_channels * 2, hidden_channels * 4)
        self.cga4 = CGABlock(hidden_channels * 4, hidden_channels * 4)
        # self.cga5 = CGABlock(hidden_channels * 4, hidden_channels * 4)
        # self.cga6 = CGABlock(hidden_channels * 4, hidden_channels * 2)

        # Output layer
        self.norm = nn.LayerNorm(hidden_channels * 4 * 8)
        self.dropout = nn.Dropout(self.dropout)
        self.head = nn.Linear(hidden_channels * 4 * 8, out_channels)


    def forward(self, x):
        # x: [B, V, C]

        x = x.permute(0, 2, 1).contiguous()  # [B, C, V]
        x = self.fc(x)  # [B, hidden_channels, V]

        # CGA blocks
        x = self.cga1(x, self.A_static)
        x = self.cga2(x, self.A_static)
        x = self.cga3(x, self.A_static)
        x = self.cga4(x, self.A_static)
        # x = self.cga5(x, self.A_static)
        # x = self.cga6(x, self.A_static)

        # flatten & head
        x = x.flatten(1)  # [B, C*V]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.head(x)

        # 保存可视化权重（默认取每层第15个样本）
        self.vis_weights['gcn_att'] = [
            self.cga1.last_A_out.detach().cpu(),
            self.cga2.last_A_out.detach().cpu(),
            self.cga3.last_A_out.detach().cpu(),
            self.cga4.last_A_out.detach().cpu()
        ]
        self.vis_weights['sa_att'] = [
            self.cga1.last_att.detach().cpu(),
            self.cga2.last_att.detach().cpu(),
            self.cga3.last_att.detach().cpu(),
            self.cga4.last_att.detach().cpu()
        ]

        return x


    def _expand_edge_index(self, edge_index, batch_size, num_nodes):
        row, col = edge_index
        edge_index_expanded = []
        for i in range(batch_size):
            offset = i * num_nodes
            edge_index_expanded.append(torch.stack([row + offset, col + offset], dim=0))
        return torch.cat(edge_index_expanded, dim=1)



class BasicGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, dropout=None):
        super(BasicGCN, self).__init__()
        self.dropout = dropout
        self.edge_index = edge_index  # [2, num_edges]

        self.gc1 = GCNConv(in_channels, hidden_channels)
        self.gc2 = GCNConv(hidden_channels, hidden_channels*2)
        self.gc3 = GCNConv(hidden_channels*2, hidden_channels*4)
        self.gc4 = GCNConv(hidden_channels*4, hidden_channels*4)
        self.head = Linear(hidden_channels *4 * 8, out_channels)  # 8 nodes × hidden

    def forward(self, x):
        # x: [B, 8, 1]
        B, N, C = x.size()
        x = x.view(B * N, C)  # [B×8, 1]

        # 拓展 edge_index 到 batch 维度
        edge_index = self.edge_index
        edge_index = self._expand_edge_index(edge_index, batch_size=B, num_nodes=N).to(x.device)

        # batch_idx 主要用于聚合时标识样本
        batch_idx = torch.arange(B).repeat_interleave(N).to(x.device)

        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = self.gc2(x, edge_index)
        x = F.relu(x)
        x = self.gc3(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gc4(x, edge_index)
        x = F.relu(x)


        # 将 [B×8, hidden] 聚合回 [B, hidden×8]
        x = x.view(B, N * x.shape[-1])
        return self.head(x)  # [B, 3]

    def _expand_edge_index(self, edge_index, batch_size, num_nodes):
        # 将 edge_index 从 [2, E] 扩展为 [2, B×E]
        edge_index = edge_index.clone()
        row, col = edge_index
        edge_index_expanded = []

        for i in range(batch_size):
            offset = i * num_nodes
            edge_index_expanded.append(torch.stack([row + offset, col + offset], dim=0))

        return torch.cat(edge_index_expanded, dim=1)
