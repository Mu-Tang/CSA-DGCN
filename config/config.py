from src.args import get_args
import torch

import os

class Config:
    def __init__(self):
        args = get_args()
        for key, value in vars(args).items():
            setattr(self, key, value)

        if not hasattr(self, 'device'):
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Config 增加
        self.train_conditions = None # 训练工况
        self.test_conditions = None # 测试工况


        self.direction_labels  = [
            [1, 0, 0],  # 工况 1
            [1, 2, 0],  # 工况 2（Y为恒载）
            [1, 2, 0],  # 工况 3（Y为恒载）
            [0, 0, 1],  # 工况 4
            [0, 0, 1],  # 工况 5
            [0, 1, 0],  # 工况 6
            [1, 1, 0],  # 工况 7
            [1, 1, 2],  # 工况 8（Z为恒载）
            [1, 1, 0],  # 工况 9
            [0, 1, 1],  # 工况10
            [2, 1, 1],  # 工况11（X为恒载）
            [0, 1, 1],  # 工况12
            [1, 0, 1],  # 工况13
            [1, 2, 1],  # 工况14（Y为恒载）
            [1, 2, 1],  # 工况15（Y为恒载）
            [1, 0, 1],  # 工况16
            [1, 2, 1],  # 工况17（Y为恒载）
            [1, 1, 1],  # 工况18
            [1, 1, 1],  # 工况19
            [1, 1, 1],  # 工况20
            [1, 1, 1],  # 工况21
        ]


    def update_paths(self):
        # 统一获取当前工况编号（1-based）
        if hasattr(self, "test_condition"):
            self.condition_id = self.test_condition + 1
        elif self.test_conditions is not None:
            self.condition_id = self.test_conditions[0]
        elif hasattr(self, "test_index"):
            self.condition_id = self.test_index // 1110 + 1
        else:
            raise ValueError("请指定 test_condition 或 test_index")

        test_str = f"test{self.condition_id:02d}"

        self.model_path = os.path.join(self.result_dir, "model_weights", f"model_{self.condition_id}.pt")
        self.result_txt = os.path.join(self.result_dir, "results_test", f"result_{self.condition_id}.txt")
        self.figure_path = os.path.join(self.result_dir, "results_plot", f"result_plot_{self.condition_id}.png")
        self.attention_path = os.path.join(self.result_dir, "results_attention", f"attn_fold{self.condition_id}.png")

        for path in [self.model_path, self.result_txt, self.figure_path, self.attention_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
