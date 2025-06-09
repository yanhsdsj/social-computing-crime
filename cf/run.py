import torch
import numpy as np
import os

from model.graph_times_net import GraphTimesNet  # 实际是 GraphTimesNet 封装
from supervisor import GraphTimesSupervisor
from lib.graph_utils import load_graph_data
from lib.data_utils import load_dataset
from save_predictions import save_predictions_to_csv, plot_training_loss


def run(
    device="cuda",
    # data_dir="data/ch",
    # data_dir="data/la",
    data_dir="data/sh",
    # adj_pkl="data/adj_mx_chicago.pkl",
    # adj_pkl="data/adj_mx_la.pkl",
    adj_pkl="data/adj_mx_shanghai07.pkl",
    output_dir="result/GraphTimesNet",
    input_dim=8,
    hidden_dim=64,
    # seq_len=8,
    seq_len=9,
    horizon=1,
    batch_size=8,
    max_epochs=200,
    patience=20,
    lr=0.001,
    use_inception=True,
):
    # Step 1: 设备设置
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Step 2: 加载图结构
    _, _, adj_mx = load_graph_data(adj_pkl)
    supports = [torch.tensor(adj_mx, dtype=torch.float32, device=device)]

    num_nodes = supports[0].shape[0]
    print(f"support[0] shape: {supports[0].shape}")  # (77,77)

    # Step 3: 加载数据集
    data = load_dataset(data_dir, batch_size)
    train_loader = data[
        "train_loader"
    ]  # x:(8,8,77,8) (B,context_point,ids,dim) y:(8,1,77,8)
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    scaler = data["scaler"]

    # for x, y in train_loader:
        # print("x shape:", x.shape)
        # print("y shape:", y.shape)
        # break

    # Step 4: 初始化模型
    model = GraphTimesNet(
        supports=supports,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        horizon=horizon,
        use_inception=use_inception,
    )
    # print("num_nodes:", num_nodes)
    # print("input_dim:", input_dim)
    # print("hidden_dim:", hidden_dim)

    # Step 5: 初始化训练器
    supervisor = GraphTimesSupervisor(
        model=model,
        adj_mx=adj_mx,
        device=device,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        output_dir=output_dir,
        log_fn=print,
    )

    # Step 6: 开始训练
    supervisor.train()
    save_predictions_to_csv(output_dir)
    plot_training_loss(supervisor, os.path.join(output_dir, "output"))

if __name__ == "__main__":
    run()
