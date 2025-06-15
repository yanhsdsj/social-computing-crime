import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# 配置参数
INPUT_FILE = 'data/ShanghaiCrimeProcessedFormat02.csv'
CRIME_TYPE_MAPPING = {
    "盗窃": ["盗窃", "盗窃罪", "盗窃案", "盗窃一审刑事判决书", "盗窃罪,掩饰、隐瞒犯罪所得罪", "盗窃罪，故意伤害罪", "盗窃罪、偷越国（边）境罪", "盗窃罪、窝藏罪", "盗窃罪、信用卡诈骗罪", "盗窃罪、掩饰、隐瞒犯罪所得罪", "盗窃、信用卡诈骗", "盗窃、诈骗罪", "盗窃罪、脱逃罪", "盗窃罪、抢劫罪", "盗窃罪、敲诈勒索罪", "盗窃罪、诈骗罪", "盗窃罪，偷越国（边）境罪", "盗窃罪、危险驾驶罪", "盗窃罪和交通肇事罪"],
    "抢劫": ["抢劫罪", "抢劫", "抢劫案", "抢劫罪、盗窃罪", "抢劫罪、贩卖毒品罪", "抢劫罪、抢夺罪", "抢劫罪、诈骗罪", "抢劫罪（最终认定为盗窃罪）", "抢劫罪、盗窃罪、交通肇事罪", "抢劫罪、寻衅滋事罪", "抢劫罪、强制猥亵罪", "抢劫罪、故意伤害罪", "抢劫罪、强奸罪", "抢劫罪，非法拘禁罪", "抢劫罪、非法拘禁罪", "抢劫罪、盗窃罪、掩饰、隐瞒犯罪所得罪", "抢劫罪、敲诈勒索罪"],
    "诈骗": ["诈骗", "诈骗罪", "信用卡诈骗", "合同诈骗罪"],
    "故意伤害": ["故意伤害", "故意伤害罪", "抢劫罪、故意伤害罪"],
    "危险驾驶": ["危险驾驶", "危险驾驶罪"],
    "寻衅滋事": ["寻衅滋事", "寻衅滋事罪", "寻衅滋事一审刑事判决书", "寻衅滋事案", "寻衅滋事罪、盗窃罪"],
    "交通肇事": ["交通肇事罪"],
    "贩卖毒品": ["贩卖毒品", "贩卖毒品罪", "运输毒品罪", "非法持有毒品罪"],
    "强奸": ["强奸罪", "强奸罪、抢劫罪", "抢劫罪、强奸罪"],
}

# 加载数据
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', header=None)
        df.columns = ["case_number", "case_type", "city", "latitude", "longitude", "court_name", "incident_time", "judgment_date"]
        df['incident_time'] = pd.to_datetime(df['incident_time'], errors='coerce')
        df = df.dropna(subset=['incident_time'])
        return df
    except FileNotFoundError:
        print(f"文件未找到，请检查路径：{file_path}")
        return None
    except Exception as e:
        print(f"读取或处理数据时出错：{e}")
        return None

# 犯罪类型归一化
def normalize_crime_type(case_type: str) -> str:
    for main_type, variants in CRIME_TYPE_MAPPING.items():
        if case_type.strip() in variants:
            return main_type
    return "其他"

# 绘制可视化图表
def visualize_crime_data(df: pd.DataFrame):
    if df is None or df.empty:
        print("没有有效的数据用于可视化")
        return

    # 犯罪类型归一化
    df['crime_type'] = df['case_type'].apply(normalize_crime_type)

    # 筛选：保留主要犯罪类型（去除“其他”和“交通肇事”）
    df = df[~df['crime_type'].isin(["其他", "交通肇事"])]

    # 图表配色方案
    colors = ['#8FDBF3', '#98F4E0', '#86E2DC', '#BEEFBF', '#E1CAF7', '#F1CFD1', '#FFDD7F', '#F9F9CA']

    # 创建包含两个子图的图表
    fig = plt.figure(figsize=(25, 10))
    plt.rcParams['font.family'] = ['SimHei']  # 设置中文字体

    # 调整子图之间的间距和顶部底部的空白
    plt.subplots_adjust(wspace=-0.1, top=0.85, bottom=0.15)  # 减少子图之间的横向距离，增加顶部和底部的空白

    # 图A：横着的柱状图（8个犯罪类型数量）
    plt.subplot(1, 2, 1)
    crime_counts = df['crime_type'].value_counts().head(8)
    bars = plt.barh(crime_counts.index, crime_counts.values, color=colors[:8])

    plt.title('上海市主要犯罪类型数量统计', fontsize=25)  # 设置子图A的标题，字体大小为25
    plt.xlabel('案件数量', fontsize=18)  # 设置X轴标签，字体大小为16
    plt.ylabel('犯罪类型', fontsize=18)  # 设置Y轴标签，字体大小为16
    plt.grid(axis='x', linestyle='--', alpha=0.7)  # 设置X轴网格线，样式为虚线，透明度为0.7

    # 设置坐标轴刻度标签的字体大小
    plt.tick_params(axis='both', which='major', labelsize=16)

    # 在柱状图旁边显示数值
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 100, bar.get_y() + bar.get_height() / 2, f'{width}', ha='left', va='center')

    # 图B：饼图（去除“盗窃”后的7类犯罪类型分布）
    plt.subplot(1, 2, 2)
    df_without_theft = df[df['crime_type'] != '盗窃']
    crime_proportions = df_without_theft['crime_type'].value_counts(normalize=True)
    proportions = crime_proportions.values
    labels = crime_proportions.index

    # 创建饼图
    wedges, texts = plt.pie(
        proportions,
        labels=None,  # 隐藏饼图上的标签
        startangle=140,
        colors=colors[1:8]
    )

    plt.title('上海市犯罪类型分布（去除盗窃）', fontsize=25)  # 设置子图B的标题，字体大小为25
    plt.axis('equal')  # 保证饼图为圆形

    # 在空白处显示图例（颜色与标签及百分比的对应关系）
    legend_labels = [f'{label}: {prop * 100:.1f}%' for label, prop in zip(labels, proportions)]
    plt.legend(wedges, legend_labels, loc='upper right', bbox_to_anchor=(1.05, 1), fontsize=16)  # 设置图例位置和字体大小

    # 设置坐标轴刻度标签的字体大小
    plt.tick_params(axis='both', which='major', labelsize=14)

    # plt.tight_layout()
    plt.savefig('imgs/crime_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主程序
df = load_data(INPUT_FILE)
if df is not None and not df.empty:
    visualize_crime_data(df)