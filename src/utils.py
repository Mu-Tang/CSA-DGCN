import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import os

def mse_loss(pred, target):
    """
    基础 MSE 损失函数
    """
    return F.mse_loss(pred, target)


def soft_loss(pred, soft_target, weight=0.2):
    """
    软标签损失项：λ * MSE(pred, soft_target)
    """
    return weight * F.mse_loss(pred, soft_target)



def plot_heatmap_weights(A_list, SA_list, save_path, condition_id):
    os.makedirs(save_path, exist_ok=True)

    for i in range(4):
        # 第15个样本（索引14），shape: [C, V, V]
        A = A_list[i][14].detach().cpu()
        SA = SA_list[i][14].detach().cpu()

        # === 差异性最大的通道（基于结构差异） ===
        diff_per_channel = ((A - SA)**2).flatten(1).mean(1)
        top_c = diff_per_channel.argmax().item()

        A_vis = A[top_c].numpy()
        SA_vis = SA[top_c].numpy()

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # === GCN 热力图 ===
        gcn = sns.heatmap(
            A_vis, ax=axs[0], square=True, cbar=True, cmap='YlOrRd'
        )
        axs[0].set_title(f'GCN Weights - Layer {i+1}', fontsize=12)
        axs[0].tick_params(labelsize=9)

        for y in range(A_vis.shape[0]):
            for x in range(A_vis.shape[1]):
                val = A_vis[y, x]
                threshold = np.median(A_vis)
                text_color = "white" if val > threshold else "black"
                # text_color = "white" if val > 1.0 else "black"
                axs[0].text(x + 0.5, y + 0.5, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=10)

        # === SA 热力图 ===
        sa = sns.heatmap(
            SA_vis, ax=axs[1], square=True, cbar=True, cmap='YlGn'
        )
        axs[1].set_title(f'SA Weights - Layer {i+1}', fontsize=12)
        axs[1].tick_params(labelsize=9)

        for y in range(SA_vis.shape[0]):
            for x in range(SA_vis.shape[1]):
                val = SA_vis[y, x]
                threshold = np.median(SA_vis)
                text_color = "white" if val > threshold else "black"
                axs[1].text(x + 0.5, y + 0.5, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'cond_{condition_id}_layer_{i+1}_weights.png'), dpi=300)
        plt.close()




def smoothed_loss(beta=0.9):
    """
    平滑指标工具（如需未来使用移动平均 loss 曲线）
    """
    class SmoothedLoss:
        def __init__(self):
            self.beta = beta
            self.avg = None

        def update(self, val):
            if self.avg is None:
                self.avg = val
            else:
                self.avg = self.beta * self.avg + (1 - self.beta) * val
            return self.avg

    return SmoothedLoss()

def collate_fn(batch):
    """
    自定义 collate 函数用于标准 PyTorch Dataloader
    输入 batch 是列表，每个元素是一个 tuple:
        (x: [T, V, C], y: [3], direction: [3], stroke_id: int, cond_id: int)
    返回：
        x_batch: [B, T, V, C]
        y_batch: [B, 3]
        direction_batch: [B, 3]
        stroke_id_batch: [B]
        cond_id_batch: [B]
    """
    xs, ys, directions, stroke_ids, cond_ids = zip(*batch)

    x_batch = torch.stack(xs, dim=0)
    y_batch = torch.stack(ys, dim=0)
    direction_batch = torch.stack(directions, dim=0)
    stroke_id_batch = torch.tensor(stroke_ids, dtype=torch.long)
    cond_id_batch = torch.tensor(cond_ids, dtype=torch.long)

    return x_batch, y_batch, direction_batch, stroke_id_batch, cond_id_batch
