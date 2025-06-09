import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import pickle

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

        # 二元交叉熵损失函数
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_loss_history = []
        self.val_loss_history = []

        os.makedirs(output_dir, exist_ok=True)

    def train(self):
        best_val_loss = float("inf")
        wait = 0

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            # 单轮训练
            train_loss = self._run_epoch(self.train_loader, train=True)
            # 单轮验证
            val_loss = self._run_epoch(self.val_loader, train=False)
            t1 = time.time()

            self.log_fn(
                f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {t1 - t0:.1f}s"
            )
            
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                self.save_model()
                self.log_fn("Saved new best model.")
            else:
                wait += 1
                if wait >= self.patience:
                    self.log_fn("Early stopping.")
                    break

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

        y_pred_bin = (y_pred >= 0.5).astype(np.float32)  # 将预测值转换为0/1

        y_pred_flat = y_pred_bin.reshape(-1)
        y_true_flat = y_true.reshape(-1)

        macro_f1 = f1_score(y_true_flat, y_pred_flat, average="macro")
        micro_f1 = f1_score(y_true_flat, y_pred_flat, average="micro")
        precision = precision_score(y_true_flat, y_pred_flat, average="macro")
        recall = recall_score(y_true_flat, y_pred_flat, average="macro")

        self.log_fn(f"[Test] Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        # 打印每个类别的详细指标
        report = classification_report(y_true_flat, y_pred_flat, target_names=[f"Class {i}" for i in range(int(y_true_flat.max()) + 1)], output_dict=True)
        for class_name, metrics in report.items():
            if class_name.isdigit():
                self.log_fn(f"Class {class_name}: Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1-score']:.4f}")

        # 保存预测结果
        pickle.dump(y_true, open(os.path.join(self.output_dir, "labels.pkl"), "wb"))
        pickle.dump(y_pred_bin, open(os.path.join(self.output_dir, "predict.pkl"), "wb"))
        # save_predictions_to_csv(self.output_dir)

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model.pt"))

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "best_model.pt")))