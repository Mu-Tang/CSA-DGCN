from config.config import Config
from feeders.feeder import StrainDataset, StrainDataLoader
from model.gcn import BasicGCN, CSADGCN
from src.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from graph.graph_phy import build_static_edge_index, build_static_adjacency
import torch
import numpy as np
import os
import sys


def main():
    print("==========开始 21 折训练（BasicGCN）==========")
    all_best_metrics = []


    for i in range(21):
        print(f"\n==== 工况 {i + 1}/21 ====")
        config = Config()
        config.test_condition = i  # 使用 test_condition 而不是 test_index
        config.result_dir = "results"
        config.update_paths()      # 确保 config.condition_id 正确
        A_static = build_static_adjacency().to(config.device)  # [8,8]
        edge_index = build_static_edge_index().to(config.device)

        # 构建数据集
        dataset = StrainDataset(config.data_paths)
        dataloader = StrainDataLoader(dataset, config)

        # 初始化模型
        # model = BasicGCN(
        #     in_channels=2,
        #     hidden_channels=32,
        #     out_channels=3,
        #     edge_index=edge_index,
        #     dropout = 0.0
        # ).to(config.device)

        model = CSADGCN(
            in_channels=2,
            hidden_channels=64,
            out_channels=3,  # x/y/z 三个载荷
            edge_index=edge_index,
            A_static=A_static,
            dropout=0.0
        ).to(config.device)

        # 日志记录
        writer = SummaryWriter(log_dir=os.path.join(config.result_dir, f"logs/fold_{i+1}"))

        # 训练器
        trainer = Trainer(config=config, dataloader=dataloader)
        best_metrics = trainer.train(model, dataloader.train_loader, dataloader.test_loader, writer=writer)
        all_best_metrics.append(best_metrics)

        writer.close()

    # 汇总结果
    print("\n==========训练完成，汇总指标==========")
    print("===== 每折最佳指标（21 折）=====")
    for i, m in enumerate(all_best_metrics):
        print(f"Fold {i + 1}:")
        print(f"  MAPE:  {m['MAPE (%)']}")
        print(f"  NRMSE: {m['NRMSE (%)']}")
        print(f"  R2:    {m['R^2']}")

    mape_all = np.array([m['MAPE (%)'] for m in all_best_metrics])
    nrmse_all = np.array([m['NRMSE (%)'] for m in all_best_metrics])
    r2_all = np.array([m['R^2'] for m in all_best_metrics])

    print("===== 三方向平均指标（±标准差）=====")
    print("MAPE:", np.mean(mape_all, axis=0), "+/-", np.std(mape_all, axis=0))
    print("NRMSE:", np.mean(nrmse_all, axis=0), "+/-", np.std(nrmse_all, axis=0))
    print("R2:", np.mean(r2_all, axis=0), "+/-", np.std(r2_all, axis=0))

def cus_val(selected_conditions):

    print(f"==========开始 {len(selected_conditions)} 折训练（BasicGCN）==========")
    all_best_metrics = []

    for fold_id, cond in enumerate(selected_conditions):
        print(f"\n==== 工况 {cond + 1}（Fold {fold_id + 1}/{len(selected_conditions)}） ====")

        config = Config()
        config.test_condition = cond
        config.result_dir = "results"
        config.update_paths()  # 会自动更新 condition_id = test_condition + 1
        edge_index = build_static_edge_index().to(config.device)
        A_static = build_static_adjacency().to(config.device)  # [8,8]

        dataset = StrainDataset(config.data_paths)
        dataloader = StrainDataLoader(dataset, config)

        # model = BasicGCN(
        #     in_channels=2,
        #     hidden_channels=32,
        #     out_channels=3,
        #     edge_index=edge_index,
        #     dropout = 0.0
        # ).to(config.device)

        model = CSADGCN(
            in_channels=2,
            hidden_channels=64,
            out_channels=3,  # x/y/z 三个载荷
            edge_index=edge_index,
            A_static=A_static,
            dropout=0.0
        ).to(config.device)

        writer = SummaryWriter(log_dir=os.path.join(config.result_dir, f"logs/fold_{cond+1:02d}"))
        trainer = Trainer(config=config, dataloader=dataloader)
        best_metrics = trainer.train(model, dataloader.train_loader, dataloader.test_loader, writer=writer)
        all_best_metrics.append(best_metrics)
        writer.close()

    # 汇总结果
    print("\n==========训练完成，汇总指标==========")
    for i, cond in enumerate(selected_conditions):
        print(f"Fold {i + 1}（工况 {cond + 1}）:")
        print(f"  MAPE:  {all_best_metrics[i]['MAPE (%)']}")
        print(f"  NRMSE: {all_best_metrics[i]['NRMSE (%)']}")
        print(f"  R2:    {all_best_metrics[i]['R^2']}")

    def masked_mean_std(arr):
        arr = np.array(arr)
        mask = arr != 0
        mean = np.sum(arr * mask, axis=0) / np.sum(mask, axis=0)
        std = np.sqrt(np.sum(((arr - mean) * mask) ** 2, axis=0) / np.sum(mask, axis=0))
        return mean, std

    # 计算
    mape_all = np.array([m['MAPE (%)'] for m in all_best_metrics])
    nrmse_all = np.array([m['NRMSE (%)'] for m in all_best_metrics])
    r2_all = np.array([m['R^2'] for m in all_best_metrics])

    # 使用 masked 版本
    mape_mean, mape_std = masked_mean_std(mape_all)
    nrmse_mean, nrmse_std = masked_mean_std(nrmse_all)
    r2_mean, r2_std = masked_mean_std(r2_all)

    # 打印
    print("===== 三方向平均指标（±标准差）=====")
    print("MAPE:", mape_mean, "+/-", mape_std)
    print("NRMSE:", nrmse_mean, "+/-", nrmse_std)
    print("R2:", r2_mean, "+/-", r2_std)

class DualLogger:
    """同时输出到控制台和文件"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    sys.stdout = DualLogger("results/run_log.txt")
    # main()
    cus_val([0,3,4,5,6,8,9,11,12,15,17,18,19,20])
