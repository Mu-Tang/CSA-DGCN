import torch
import numpy as np
import os
from sklearn.metrics import r2_score
from src.fig import plot_figure
from src.utils import plot_heatmap_weights


class Trainer:
    def __init__(self, config, dataloader):
        self.config = config
        self.dataloader = dataloader
        self.criterion = torch.nn.MSELoss()

    def train(self, model, train_loader, test_loader, writer=None):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        best_loss = float('inf')
        best_metrics = None

        for epoch in range(self.config.epochs):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                pred = model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            val_loss, val_preds, val_targets = self._validate(model, test_loader, return_preds=True)

            scheduler.step()
            # === 写入 TensorBoard ===
            if writer is not None:
                writer.add_scalar("train/loss", avg_loss, epoch)
                writer.add_scalar("val/loss", val_loss, epoch)

                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("train/lr", current_lr, epoch)

                metrics = self._compute_metrics(val_preds, val_targets)

                # 把 x/y/z 的值写到同一个 tag 下的不同 scalar_name
                writer.add_scalars(f"fold_{self.config.condition_id:02d}/val_MAPE", {
                    'x': metrics["MAPE (%)"][0],
                    'y': metrics["MAPE (%)"][1],
                    'z': metrics["MAPE (%)"][2]
                }, epoch)

                writer.add_scalars(f"fold_{self.config.condition_id:02d}/val_NRMSE", {
                    'x': metrics["NRMSE (%)"][0],
                    'y': metrics["NRMSE (%)"][1],
                    'z': metrics["NRMSE (%)"][2]
                }, epoch)

                writer.add_scalars(f"fold_{self.config.condition_id:02d}/val_R2", {
                    'x': metrics["R^2"][0],
                    'y': metrics["R^2"][1],
                    'z': metrics["R^2"][2]
                }, epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                best_metrics = self._compute_metrics(val_preds, val_targets)
                self._save_best_results(model, val_preds, val_targets)
                torch.save(model.state_dict(), self.config.model_path)

            print(f"[Epoch {epoch+1}/{self.config.epochs}] Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

        return best_metrics

    def _validate(self, model, val_loader, return_preds=False):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                output = model(x)
                preds.append(output.cpu().numpy())
                targets.append(y.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        preds_inv = self.dataloader.inverse_transform(preds)
        targets_inv = self.dataloader.inverse_transform(targets)

        # === 使用 MAPE + NRMSE 作为验证指标（只考虑有效方向）===
        metrics = self._compute_metrics(preds_inv, targets_inv)
        valid_mape = [v for v in metrics["MAPE (%)"] if v > 1e-3]
        valid_nrmse = [v for v in metrics["NRMSE (%)"] if v > 1e-3]

        mape = np.mean(valid_mape) if valid_mape else 0.0
        nrmse = np.mean(valid_nrmse) if valid_nrmse else 0.0
        val_loss = mape + nrmse

        if return_preds:
            return val_loss, preds_inv, targets_inv
        return val_loss

    def _compute_metrics(self, preds, targets, eps=1e-8):
        preds = preds.reshape(-1, 3)
        targets = targets.reshape(-1, 3)

        metrics = {"MAPE (%)": [], "NRMSE (%)": [], "R^2": []}
        for i in range(3):
            p, t = preds[:, i], targets[:, i]
            mask = np.abs(t) > 1e-2

            if np.any(mask):
                # MAPE
                mape = np.mean(np.abs((p[mask] - t[mask]) / (t[mask] + eps))) * 100

                # NRMSE
                rmse = np.sqrt(np.mean((p[mask] - t[mask]) ** 2))
                mean_t = np.mean(np.abs(t[mask]))
                nrmse = rmse / (mean_t + eps) * 100 if mean_t > eps else 0.0

                # R²
                ss_total = np.sum((t[mask] - np.mean(t[mask])) ** 2)
                ss_res = np.sum((t[mask] - p[mask]) ** 2)
                r2 = 1 - ss_res / (ss_total + eps) if ss_total > eps else 0.0
            else:
                mape = nrmse = r2 = 0.0  # or np.nan

            metrics["MAPE (%)"].append(round(mape, 4))
            metrics["NRMSE (%)"].append(round(nrmse, 4))
            metrics["R^2"].append(round(r2, 4))

        return metrics

    def _save_best_results(self, model, preds, targets):
        fold_id = self.config.condition_id
        base_path = self.config.result_dir
        os.makedirs(os.path.join(base_path, "results_plot"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "results_attention"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "results_test"), exist_ok=True)

        plot_figure(save_path=os.path.join(base_path, "results_plot"),
                    label=targets, pred=preds, title_suffix=str(fold_id))

        plot_heatmap_weights(
            model.vis_weights['gcn_att'],
            model.vis_weights['sa_att'],
            save_path=os.path.join(base_path, "results_attention"),
            condition_id=fold_id
        )

        self._save_predictions_txt(preds, targets,
                                   os.path.join(base_path, "results_test", f"result_{fold_id}.txt"))

    def _save_predictions_txt(self, preds, targets, path):
        metrics = self._compute_metrics(preds, targets)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding='utf-8') as f:
            f.write("预测值\t\t实际值\n")
            for pt, tt in zip(preds, targets):
                p_str = "[" + ", ".join(f"{v:.6f}" for v in pt) + "]"
                t_str = "[" + ", ".join(f"{v:.6f}" for v in tt) + "]"
                f.write(f"{p_str}\t{t_str}\n")

            f.write("\n评估指标：\n")
            f.write("MAPE:  [" + ", ".join(f"{v:.2f}%" for v in metrics['MAPE (%)']) + "]\n")
            f.write("NRMSE: [" + ", ".join(f"{v:.2f}%" for v in metrics['NRMSE (%)']) + "]\n")
            f.write("R2:    [" + ", ".join(f"{v:.3f}" for v in metrics['R^2']) + "]\n")
