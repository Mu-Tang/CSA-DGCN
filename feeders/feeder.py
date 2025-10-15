import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset

class StrainDataset(Dataset):
    def __init__(self, data_paths):
        """
        每个 CSV 表示一个行程，按工况顺序排列，每工况370条。
        构建 shape = [3行程, 21工况, 370条, 8节点, 1特征]
        """
        self.X_all = []
        self.Y_all = []

        for path in data_paths:
            df = pd.read_csv(path, header=0)
            strain = df.iloc[:, 2:10].values.reshape(21, 370, 8, 1)
            load = df.iloc[:, 10:13].values.reshape(21, 370, 3)
            self.X_all.append(strain)
            self.Y_all.append(load)

        self.X_all = np.stack(self.X_all, axis=0)  # [3, 21, 370, 8, 1]
        self.Y_all = np.stack(self.Y_all, axis=0)  # [3, 21, 370, 3]

    def get_split_data(self, test_condition):
        """
        返回当前工况作为测试，其余为训练的数据。
        test_condition: int，0-based，范围 [0 ~ 20]
        """
        X_train, Y_train, X_test, Y_test = [], [], [], []

        for stroke in range(self.X_all.shape[0]):  # 遍历3个行程
            for cond in range(21):
                # x = self.X_all[stroke, cond]  # [370, 8, 1]
                y = self.Y_all[stroke, cond]  # [370, 3]

                # 原始应变数据
                x_base = self.X_all[stroke, cond]  # [370, 8, 1]
                # 构造辅助特征通道（用行程编号归一化为 0/0.5/1）
                stroke_value = stroke / 2.0
                aux_feat = np.full_like(x_base, stroke_value)  # [370, 8, 1]
                # 广播拼接第2通道，得到 [370, 8, 2]
                x = np.concatenate([x_base, aux_feat], axis=-1)

                if cond == test_condition:
                    X_test.append(x)
                    Y_test.append(y)
                else:
                    X_train.append(x)
                    Y_train.append(y)

        return (
            np.concatenate(X_train, axis=0),  # [B_train, 8, 1]
            np.concatenate(Y_train, axis=0),  # [B_train, 3]
            np.concatenate(X_test, axis=0),   # [1110, 8, 1]
            np.concatenate(Y_test, axis=0)    # [1110, 3]
        )

class StrainDataLoader:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.train_loader, self.test_loader = self.split_and_load()

    def split_and_load(self):
        # 当前测试工况编号（0-based）
        test_cond = self.config.test_condition
        self.config.condition_id = test_cond + 1  # 1-based 记录当前工况编号

        X_train, Y_train, X_test, Y_test = self.dataset.get_split_data(test_cond)

        # === X归一化（基于训练集） ===
        self.X_scaler = MinMaxScaler()
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_test_flat = X_test.reshape(len(X_test), -1)
        X_train_scaled = self.X_scaler.fit_transform(X_train_flat).reshape(len(X_train), 8, 2)
        X_test_scaled = self.X_scaler.transform(X_test_flat).reshape(len(X_test), 8, 2)

        # === Y归一化（基于训练集） ===
        self.Y_scaler = MinMaxScaler()
        Y_train_scaled = self.Y_scaler.fit_transform(Y_train)
        Y_test_scaled = self.Y_scaler.transform(Y_test)
        self.Y_test_original = Y_test

        # === 转Tensor ===
        X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.config.device)
        X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.config.device)
        Y_train = torch.tensor(Y_train_scaled, dtype=torch.float32).to(self.config.device)
        Y_test = torch.tensor(Y_test_scaled, dtype=torch.float32).to(self.config.device)

        train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=self.config.batch_size, shuffle=False)
        return train_loader, test_loader

    def inverse_transform(self, Y_scaled):
        return self.Y_scaler.inverse_transform(Y_scaled)
