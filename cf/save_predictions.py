import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from supervisor import GraphTimesSupervisor


def save_predictions_to_csv(output_dir="./result/output"):
    """
    从保存的pickle文件中加载预测结果和真实标签，并将其保存为CSV文件
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载保存的预测结果和标签
        y_true = pickle.load(open(os.path.join(output_dir, "labels.pkl"), "rb"))
        y_pred = pickle.load(open(os.path.join(output_dir, "predict.pkl"), "rb"))
        
        # 展平数组
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'True_Label': y_true_flat,
            'Predicted_Label': y_pred_flat,
            'Predicted_Probability': y_pred.reshape(-1)  # 如果需要概率值
        })
        
        # 保存为CSV
        csv_path = os.path.join(output_dir, "predict.csv")
        df.to_csv(csv_path, index=False)
        print(f"预测结果已保存到: {csv_path}")
        
        return True
        
    except Exception as e:
        print(f"保存预测结果时出错: {e}")
        return False

def plot_training_loss(supervisor: GraphTimesSupervisor, output_dir="./result/output"):
    """
    绘制并保存训练损失下降图
    
    参数:
        supervisor: 训练完成的GraphTimesSupervisor实例
        output_dir: 输出目录，默认为"./result/output"
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 从supervisor中获取训练历史记录
        # 注意：这里需要你的supervisor类有记录训练历史的属性
        # 如果没有，你需要修改supervisor类来记录这些信息
        
        # 假设supervisor有train_loss_history和val_loss_history属性
        train_loss = supervisor.train_loss_history
        val_loss = supervisor.val_loss_history
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training Loss', color='blue')
        plt.plot(val_loss, label='Validation Loss', color='red')
        
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plot_path = os.path.join(output_dir, "training_loss.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练损失图已保存到: {plot_path}")
        
        return True
        
    except Exception as e:
        print(f"生成训练损失图时出错: {e}")
        return False

if __name__ == "__main__":
    save_predictions_to_csv()