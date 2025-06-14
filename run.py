import torch
import numpy as np
import os

from model.graph_times_net import GraphTimesNet
from supervisor import GraphTimesSupervisor
from lib.graph_utils import load_graph_data
from lib.data_utils import load_dataset
from save_predictions import save_predictions_to_csv, plot_training_loss


def run(
    device="cuda",
    data_dir="data/ch",
    # data_dir="data/la",
    # data_dir="data/sh",
    adj_pkl="data/adj_mx_chicago.pkl",
    # adj_pkl="data/adj_mx_la.pkl",
    # adj_pkl="data/adj_mx_shanghai07.pkl",
    output_dir_base="test_with_v2_result",
    input_dim=8,
    hidden_dim=64,
    seq_len=8,
    # seq_len=9,
    horizon=1,
    batch_size=8,
    max_epochs=200,
    patience=30,
    lr=0.002,
    use_inception=True,
):
    # Step 1: 设备设置
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Step 2: 获取数据集名称
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    
    # Step 3: 设置输出目录
    model_output_dir = os.path.join(output_dir_base, f"GraphTimesNet_{dataset_name}")
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir_base, "output"), exist_ok=True)

    # Step 4: 加载图结构
    _, _, adj_mx = load_graph_data(adj_pkl)
    supports = [torch.tensor(adj_mx, dtype=torch.float32, device=device)]

    num_nodes = supports[0].shape[0]
    print(f"support[0] shape: {supports[0].shape}")  # (77,77)

    # Step 5: 加载数据集
    data = load_dataset(data_dir, batch_size)
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    scaler = data["scaler"]

    # Step 6: 初始化模型
    model = GraphTimesNet(
        supports=supports,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        horizon=horizon,
        use_inception=use_inception,
    )

    # pre 7：在初始化 supervisor 之前创建日志目录
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 定义一个同时打印和记录到文件的函数
    def log_fn(message):
        print(message)
        with open(os.path.join(model_output_dir, "console_log.txt"), 'a') as f:
            f.write(message + "\n")
    

    # Step 7: 初始化训练器
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
        output_dir=model_output_dir,
        log_fn=log_fn,
        dataset_name=dataset_name, 
    )

    # Step 8: 开始训练
    supervisor.train()
    save_predictions_to_csv(model_output_dir, os.path.join(output_dir_base, "output"), dataset_name)
    plot_training_loss(supervisor, os.path.join(output_dir_base, "output"), dataset_name)

if __name__ == "__main__":
    run()