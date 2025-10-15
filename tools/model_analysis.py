import torch
from config.config import Config
from thop import profile
import time
from src.args import get_args
from graph.graph_phy import build_static_edge_index, build_static_adjacency


# 模型导入
from model.gcn import CSADGCN

config = Config()
config = get_args()
edge_index = build_static_edge_index().to(config.device)
A_static = build_static_adjacency().to(config.device)  # [8,8]
# 模型初始化参数（与你实验中保持一致）
model = CSADGCN(
    in_channels=2,
    hidden_channels=64,
    out_channels=3,  # x/y/z 三个载荷
    edge_index=edge_index,
    A_static=A_static,
    dropout=0.0
).to(config.device)

dummy_input = torch.randn(1, 8, 2).to(config.device)  # 注意输入格式是否与 forward() 匹配

# 计算 Params 和 FLOPs
macs, params = profile(model, inputs=(dummy_input, ), verbose=False)

# 推理时间估算
model.eval()
with torch.no_grad():
    start = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    end = time.time()
    inference_time = (end - start) / 100 * 1000  # 单次 ms

print(f"Params (M): {params / 1e6:.4f}")
print(f"FLOPs  (G): {macs / 1e9:.4f}")
print(f"Inference Time (ms): {inference_time:.4f}")
