import torch
import torch.nn as nn

class GraphBuilder(nn.Module):
    def __init__(self, num_nodes=8, device='cpu'):
        super().__init__()
        self.num_nodes = num_nodes
        self.device = torch.device(device)

        # 1. 定义强连接（固定）
        self.strong_dict = {
            0: [6, 4],  # ZR1S-1 → ZR1S-2, ZR2S-2
            1: [0, 2],  # ZR1S-4 → ZR1S-1, ZR2S-1
            6: [7, 5],  # ZR1S-2 → ZR1S-3, ZR2S-3
            7: [1, 3],  # ZR1S-3 → ZR1S-4, ZR2S-4

            2: [3, 1],  # ZR2S-1 → ZR2S-4, ZR1S-4
            3: [5, 7],  # ZR2S-4 → ZR2S-3, ZR1S-3
            4: [2, 0],  # ZR2S-2 → ZR2S-1, ZR1S-1
            5: [4, 6]  # ZR2S-3 → ZR2S-2, ZR1S-2
        }
        # 2. 弱连接mask (0/1)
        self.weak_mask = torch.zeros((num_nodes, num_nodes), device=self.device)

        for i in range(num_nodes):
            self.weak_mask[i, i] = 1  # 加上自环

        # 同截面
        A_sec = [0, 6, 7, 1] # ZR1S-1 → ZR1S-4
        B_sec = [2, 4, 5, 3] # ZR2S-1 → ZR2S-4
        for section in [A_sec, B_sec]:
            for i in section:
                for j in section:
                    if i != j:
                        self.weak_mask[i, j] = 1

        # 斜向弱连接
        for (i, j) in [(0, 2), (1, 3), (6, 4), (7, 5)]:
            self.weak_mask[i, j] = 1
            self.weak_mask[j, i] = 1

        # 3. 弱连接可学习权重
        self.weak_param = nn.Parameter(torch.full((num_nodes, num_nodes), 1e-6, device=self.device))

    def build_static_graph(self):
        """
        构建静态结构图（邻接矩阵 A 与边权 A_weight）
        返回：
            edge_index: [2, E] LongTensor，双向边索引
            edge_weight: [E] FloatTensor，对应每条边的权重
        """
        normal_edges = [
            (0, 6), (6, 7), (7, 1), (0, 1),
            (2, 4), (4, 5), (5, 3), (2, 3)
        ]
        diagonal_edges = [
            (0, 2), (2, 6), (6, 4), (4, 7),
            (7, 5), (5, 1), (1, 3), (3, 0)
        ]

        edge_index = []
        edge_weights = []

        for i, j in normal_edges:
            edge_index.extend([[i, j], [j, i]])
            edge_weights.extend([0.5, 0.5])

        for i, j in diagonal_edges:
            edge_index.extend([[i, j], [j, i]] * 2)
            edge_weights.extend([0.5, 0.5] * 2)

        for i in range(8):
            edge_index.append([i, i])
            edge_weights.append(1.0)

        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).T  # [2, E]
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32, device=self.device)  # [E]

        return edge_index, edge_weights

    def get_adj_matrix(self):
        edge_index, edge_weights = self.build_static_graph()  # [2, E], [E]
        num_nodes = self.num_nodes
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=self.device)

        for idx in range(edge_index.shape[1]):
            src, tgt = edge_index[0, idx], edge_index[1, idx]
            adj_matrix[src, tgt] = edge_weights[idx]

        return adj_matrix  # shape [N, N]

    def forward(self):
        # 返回强连接 dict，弱mask，弱param，静态图邻接矩阵
        return self.strong_dict, self.weak_mask.to(self.device), self.weak_param.to(self.device),None #self.get_adj_matrix()
