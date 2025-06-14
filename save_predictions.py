import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from supervisor import GraphTimesSupervisor


def save_predictions_to_csv(model_output_dir, output_dir, dataset_name):
    """
    从保存的pickle文件中加载预测结果和真实标签，并将其保存为CSV文件
    
    参数:
        model_output_dir: 模型输出目录，包含labels.pkl和predict.pkl
        output_dir: 输出目录，用于保存CSV文件
        dataset_name: 数据集名称，用于文件名
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载保存的预测结果和标签
        y_true = pickle.load(open(os.path.join(model_output_dir, "labels.pkl"), "rb"))
        y_pred = pickle.load(open(os.path.join(model_output_dir, "predict.pkl"), "rb"))
        
        # 展平数组
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'True_Label': y_true_flat,
            'Predicted_Label': y_pred_flat,
            'Predicted_Probability': y_pred.reshape(-1)
        })
        
        # 保存为CSV
        csv_path = os.path.join(output_dir, f"{dataset_name}_predict.csv")
        df.to_csv(csv_path, index=False)
        print(f"Prediction saved at: {csv_path}")
        
        return True
        
    except Exception as e:
        print(f"Error Saving Prediction: {e}")
        return False

def plot_training_loss(supervisor: GraphTimesSupervisor, output_dir, dataset_name):
    """
    绘制并保存训练损失下降图
    
    参数:
        supervisor: 训练完成的GraphTimesSupervisor实例
        output_dir: 输出目录
        dataset_name: 数据集名称，用于文件名
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 获取训练历史记录
        train_loss = supervisor.train_loss_history
        val_loss = supervisor.val_loss_history

        loss_df = pd.DataFrame({
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        csv_loss_path = os.path.join(output_dir, f"{dataset_name}_training_loss.csv")
        loss_df.to_csv(csv_loss_path, index_label='epoch')
        print(f"Training loss CSV saved at: {csv_loss_path}")
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training Loss', color='blue')
        plt.plot(val_loss, label='Validation Loss', color='red')
        
        plt.title(f'Training and Validation Loss Over Epochs ({dataset_name})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plot_path = os.path.join(output_dir, f"{dataset_name}_training_loss.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training Loss png saved at: {plot_path}")
        
        return True
        
    except Exception as e:
        print(f"Error Saving Training Loss png: {e}")
        return False

if __name__ == "__main__":
    # 示例用法
    save_predictions_to_csv("result/GraphTimesNet_la", "result/output", "la")