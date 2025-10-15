import torch
import torch.nn as nn
from graph.graph_builder import GraphBuilder
from src.gcn_layers import CustomGCNLayer

class CSA_DGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, num_heads, alpha, dropout):
        super().__init__()
        self.graph = GraphBuilder()
        strong_dict, weak_mask, weak_param, edge_weight_adj = self.graph() # 读取邻接矩阵

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        # self.cond_scale = torch.tensor(0.0)
        # self.alpha = nn.Parameter(torch.tensor(alpha))


        self.ffn_blocks = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.LayerNorm(hidden_dim)  # 可加（但需验证）
        )

        # 将 direction_label 映射为隐藏向量，用于条件调制
        self.condition_proj = nn.Linear(3, hidden_dim)

        for i in range(num_layers):
            layer = CustomGCNLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                strong_dict=strong_dict,
                weak_mask=weak_mask,
                weak_param=weak_param,
                edge_weight_adj=edge_weight_adj,  # 新增
                alpha=alpha,  # 静态图引导权重
                use_tcn=False,
                num_heads=num_heads
            )
            self.layers.append(layer)
            self.ffn_blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.norm = nn.LayerNorm(hidden_dim)
        self.linear_proj = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x):
        # x: [B, N, C]  -> N: sensor count, C: input features per node

        x = x.permute(0, 2, 1)  # [B, C, N]
        x = self.embedding(x)  # [B, hidden_dim, N]
        x = x.permute(0, 2, 1)  # [B, N, hidden_dim]


        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x)  # shape [B, N, hidden_dim]
            x = self.layer_norms[i](x)
            x = self.ffn_blocks[i](x) + residual

        x = self.norm(x)
        x = self.linear_proj(x)
        x = x.mean(dim=1)
        out = self.head(x)
        return out  # shape [B, 3] -> 3D load vector
