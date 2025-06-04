import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# 数据加载器
class TimeSeriesDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs  # [N, T, N_nodes, D]
        self.ys = ys

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        # 返回单个样本(x,y)
        x = self.xs[idx]
        y = self.ys[idx]
        # 转成torch.tensor，并确保是float32
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


# 从.npz文件加载训练、验证、测试集，构造DataLoader
def load_dataset(dataset_dir, batch_size, val_batch_size=None, test_batch_size=None):
    data = {}
    for category in ["train", "val", "test"]:
        path = os.path.join(dataset_dir, f"{category}.npz")
        npz_data = np.load(path)
        data[f"x_{category}"] = npz_data["x"]
        data[f"y_{category}"] = npz_data["y"]

    train_set = TimeSeriesDataset(data["x_train"], data["y_train"])
    val_set = TimeSeriesDataset(data["x_val"], data["y_val"])
    test_set = TimeSeriesDataset(data["x_test"], data["y_test"])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size or batch_size)
    test_loader = DataLoader(test_set, batch_size=test_batch_size or batch_size)

    class DummyScaler:
        def transform(self, x):
            return x

        def inverse_transform(self, x):
            return x

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "y_test": data["y_test"],
        "scaler": DummyScaler(),
    }
