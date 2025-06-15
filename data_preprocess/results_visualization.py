import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from matplotlib.gridspec import GridSpec

# 数据
cities = ['Chicago', 'Los Angeles', 'Shanghai']
models = ['origin', 'Crime Forecaster+']
metrics = {
    'Macro F1': [0.6640, 0.7382, 0.5354, 0.7137, 0.0816, 0.7641],
    'Micro F1': [0.6954, 0.7429, 0.6046, 0.7719, 0.0951, 0.9941],
    'Precision': [0.6869, 0.7439, 0.5609, 0.7410, 0.0610, 0.8023],
    'Recall': [0.6449, 0.7368, 0.5226, 0.7013, 0.6124, 0.7345]
}

# 创建颜色选择器
color_picker1 = widgets.ColorPicker(
    description='Color 1:',
    value='#1f77b4',
    width=100
)
color_picker2 = widgets.ColorPicker(
    description='Color 2:',
    value='#ff7f0e',
    width=100
)

# 创建图形输出区域
out = widgets.Output()


# 定义更新图表的函数
def update_chart(color1, color2):
    with out:
        out.clear_output(wait=True)
        # 调整图形大小和子图布局
        fig = plt.figure(figsize=(14, 10))  # 减小高度，增加宽度
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)  # 减小间距

        colors = [color1, color2]

        for i, metric in enumerate(metrics.keys()):
            ax = fig.add_subplot(gs[i])
            x = np.arange(len(cities))
            width = 0.35  # 恢复柱状图宽度

            # 绘制柱状图
            for j, model in enumerate(models):
                model_values = metrics[metric][j::2]
                bars = ax.bar(x + j * width, model_values, width, label=model, color=colors[j])

                # 添加数据标签（调整位置避免遮挡）
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.015,  # 微调标签位置
                            f'{height:.4f}', ha='center', va='bottom', rotation=0, fontsize=9)

            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f'{metric} Comparison', fontsize=12)
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(cities, rotation=0, fontsize=10)
            ax.legend(fontsize=9, loc='upper left')  # 调整图例位置

            # 动态设置Y轴上限，根据数据特性优化显示
            max_val = max(model_values)
            if max_val < 1:
                ax.set_ylim(0, 1.05)  # 对于F1等指标，设置固定上限
            else:
                ax.set_ylim(0, max_val * 1.1)  # 对于其他指标，动态设置上限

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为顶部留出空间
        plt.show()


# 创建交互式控件
interactive_plot = widgets.interactive(
    update_chart,
    color1=color_picker1,
    color2=color_picker2
)

# 显示控件和图表
display(widgets.HBox([color_picker1, color_picker2]))
display(interactive_plot)
display(out)