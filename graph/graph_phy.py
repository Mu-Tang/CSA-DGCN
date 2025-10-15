import numpy as np
import torch


def build_static_adjacency(num_nodes=8, connection_type='sparse'):
    """
    构建固定静态图邻接矩阵
    :param num_nodes: 节点数（8个）
    :param connection_type: 'sparse' or 'full_cross'
    :return: [num_nodes, num_nodes] 的 torch.Tensor
    """
    adj = np.zeros((num_nodes, num_nodes))

    # --- 截面内连接（ZR1S: 节点0,6,7,1） ---
    adj[0, 6] = adj[6, 0] = 1
    adj[6, 7] = adj[7, 6] = 1
    adj[7, 1] = adj[1, 7] = 1
    adj[1, 0] = adj[0, 1] = 1

    # --- 截面内连接（ZR2S: 节点2,4,5,3） ---
    adj[2, 4] = adj[4, 2] = 1
    adj[4, 5] = adj[5, 4] = 1
    adj[5, 3] = adj[3, 5] = 1
    adj[3, 2] = adj[2, 3] = 1

    # --- 跨截面连接（ sparse 对位连接） ---
    cross_pairs = [(0, 2), (2, 6), (6, 4), (4, 7),
            (7, 5), (5, 1), (1, 3), (3, 0)]
    for i, j in cross_pairs:
        adj[i, j] = adj[j, i] = 1

    # 增加自环
    for i in range(num_nodes):
        adj[i, i] = 1

    return torch.tensor(adj, dtype=torch.float32)

def build_static_edge_index():
    """
    基于静态邻接矩阵构造 PyG 格式的 edge_index
    :return: edge_index (LongTensor) [2, num_edges]
    """
    adj = build_static_adjacency()  # [8, 8]
    edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous()  # shape: [2, num_edges]
    return edge_index.long()
