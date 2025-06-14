import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import pickle
import pandas as pd
from sklearn import metrics

class GraphTimesSupervisor:
    def __init__(
        self,
        model,
        adj_mx,
        device,
        scaler,
        train_loader,
        val_loader,
        test_loader,
        lr=0.001,
        max_epochs=200,
        patience=20,
        output_dir="./result",
        log_fn=print,
        dataset_name="default",
    ):

        self.device = device
        self.model = model.to(device)
        self.scaler = scaler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs
        self.patience = patience
        self.output_dir = output_dir
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        # 添加日志文件路径
        self.log_file = os.path.join(output_dir, "training_log.txt")
        os.makedirs(output_dir, exist_ok=True)
        
        # 清空或创建日志文件
        with open(self.log_file, 'w') as f:
            f.write("===== Training Logs =====\n")


        # 二元交叉熵损失函数
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_loss_history = []
        self.val_loss_history = []

        os.makedirs(output_dir, exist_ok=True)

    def _log(self, message):
        """同时打印并记录到文件"""
        self.log_fn(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")

    def train(self):
        best_val_loss = float("inf")
        wait = 0
        epoch_times = []  # 新增：记录每个epoch的耗时
        total_time = 0    # 新增：记录总耗时

        # 记录开始时间
        self._log(f"Training Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Max epoch: {self.max_epochs}, Early Stopping patience: {self.patience}")

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            # 单轮训练
            train_loss = self._run_epoch(self.train_loader, train=True)
            # 单轮验证
            val_loss = self._run_epoch(self.val_loader, train=False)
            t1 = time.time()

            epoch_time = t1 - t0
            epoch_times.append(epoch_time)  # 新增
            total_time += epoch_time        # 新增

            self._log(
                f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s"
            )
            
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                self.save_model()
                # self.log_fn("Saved new best model.")
                self._log(f"New Best Val Loss: {best_val_loss:.4f}, save model")

            else:
                wait += 1
                if wait >= self.patience:
                    # self.log_fn("Early stopping.")
                    self._log(f"Early stopping at epoch {epoch}, Best Val loss: {best_val_loss:.4f}")

                    break
        # 训练结束记录
        self._log(f"training ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Final Best Val Loss: {best_val_loss:.4f}")
        

        # 保存每个epoch的时间和总时间到csv
        time_df = pd.DataFrame({
            'epoch': list(range(1, len(epoch_times)+1)),
            'epoch_time': epoch_times
        })
        time_df.loc['total'] = ['total', total_time]
        time_csv_path = os.path.join(self.output_dir, f"{self.dataset_name}_time.csv")
        time_df.to_csv(time_csv_path, index=False)
        self._log(f"Epoch times and total time saved at: {time_csv_path}")

        self.load_model()
        self.evaluate()

    def _run_epoch(self, loader, train=False):
        self.model.train() if train else self.model.eval()
        total_loss = 0
        for x, y in loader:
            x = x.to(self.device)  # [B, T, N, D]
            y = y.to(self.device)  # [B, 1, N, D]

            y_pred = self.model(x)  # [B, 1, N, D]
            loss = self.loss_fn(y_pred, y)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self):
        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = torch.sigmoid(self.model(x))  # 预测结果做 sigmoid
                preds.append(y_pred.cpu().numpy())
                trues.append(y.cpu().numpy())

        # 打印pred+true
        y_pred = np.concatenate(preds, axis=0)  # [B, H, N, 1]
        y_true = np.concatenate(trues, axis=0)
        # df = pd.DataFrame({
        #     "True_Label": y_true.reshape(-1),
        #     "Predicted_Label": y_pred_bin.reshape(-1),
        #     "Predicted_Probability": y_pred.reshape(-1)
        # })
        # csv_path = os.path.join(self.output_dir, f"{self.dataset_name}_predict_true.csv")
        # df.to_csv(csv_path, index=False)
        # self._log(f"Prediction and true labels saved at: {csv_path}")

        y_pred_bin = (y_pred >= 0.5).astype(np.float32)  # 将预测值转换为0/1

        y_pred_flat = y_pred_bin.reshape(-1)
        y_true_flat = y_true.reshape(-1)

        macro_f1 = f1_score(y_true_flat, y_pred_flat, average="macro")
        micro_f1 = f1_score(y_true_flat, y_pred_flat, average="micro")
        precision = precision_score(y_true_flat, y_pred_flat, average="macro")
        recall = recall_score(y_true_flat, y_pred_flat, average="macro")

        self._log(f"[Test] Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        # 打印每个类别的详细指标
        try:
            report = classification_report(y_true_flat, y_pred_flat, target_names=[f"Class {i}" for i in range(int(y_true_flat.max()) + 1)], output_dict=True)
            # 保存详细指标到文件
            metrics_file = os.path.join(self.output_dir, "class_metrics.txt")
            with open(metrics_file, 'w') as f:
                f.write("===== Detailed Class Metrics =====\n")
                for class_name, metrics in report.items():
                    if class_name.isdigit():  # 只打印类别编号部分
                        metric_str = f"Class {class_name}: Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1-score']:.4f}"
                        self._log(metric_str)
                        f.write(metric_str + "\n")
            self._log(f"Detailed class metrics saved to: {metrics_file}")
        except Exception as e:
            self._log(f"Warning: Could not print detailed classification report: {e}")

        # 保存预测结果 - 现在会自动保存到以数据集命名的目录中
        pickle.dump(y_true, open(os.path.join(self.output_dir, "labels.pkl"), "wb"))
        pickle.dump(y_pred_bin, open(os.path.join(self.output_dir, "predict.pkl"), "wb"))


    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model.pt"))

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "best_model.pt")))