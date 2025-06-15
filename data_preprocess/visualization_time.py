import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 读取CSV文件
file_path = 'data/ShanghaiCrimeProcessedFormat02.csv'


def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', header=None)
        df.columns = ["case_number", "case_type", "city", "latitude", "longitude",
                      "court_name", "incident_time", "judgment_date"]
        df['incident_time'] = pd.to_datetime(df['incident_time'], errors='coerce')
        df = df.dropna(subset=['incident_time'])
        df['month'] = df['incident_time'].dt.month
        df['year'] = df['incident_time'].dt.year
        return df
    except FileNotFoundError:
        print(f"文件未找到，请检查路径：{file_path}")
        return None
    except Exception as e:
        print(f"读取或处理数据时出错：{e}")
        return None


def plot_line_monthly(dataframe):
    if dataframe is None or dataframe.empty:
        print("没有有效的数据用于绘制折线图")
        return

    # 按年月统计犯罪数量，得到每个月的犯罪数量
    dataframe['date'] = pd.to_datetime(
        dataframe['incident_time'].dt.year.astype(str) + '-' + dataframe['incident_time'].dt.month.astype(str))
    monthly_counts = dataframe.groupby('date').size().reset_index(name='Count')

    plt.figure(figsize=(12, 6))
    plt.rcParams['font.family'] = ['SimHei']  # 设置中文字体
    plt.plot(
        monthly_counts['date'],
        monthly_counts['Count'],
        color='#5DBFE9',  # 线条颜色
        marker='o',
        markerfacecolor='#338FEF',  # 标记填充颜色
        markeredgecolor='#FFFFFF',  # 标记边框颜色
        markersize=5,  # 标记大小
        linewidth=2,
        alpha=0.8
    )

    plt.title('上海市每月犯罪数量统计（1992年-2020年1月）', fontsize=20)
    plt.xlabel('时间（以月为单位）', fontsize=14)
    plt.ylabel('犯罪数量', fontsize=14)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=16)  # 增大坐标轴刻度标签的字体大小
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('imgs/line_monthly.png', dpi=300)
    plt.show()


# 主程序
df = load_data(file_path)
if df is not None and not df.empty:
    plot_line_monthly(df)