import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
file_path = 'data/ChinaCrimeDatas.csv'
df = pd.read_csv(file_path)

# 需保留的字段
required_columns = [
    'case_number', 'case_type', 'city', 'latitude', 'longitude',
    'court_name',
    'incident_province', 'incident_city', 'incident_county',
    'incident_time', 'judgment_date'
]

# 第一步：统计所有 required_columns 中空值的比例
null_ratios = df[required_columns].isnull().mean()
print("\nNull Ratios (Before Filtering):")
print(null_ratios)

# 可视化空值比例
plt.figure(figsize=(10, 6))
null_ratios.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.ylabel('Null Ratio')
plt.title('Null Ratios of Required Columns')
plt.tight_layout()
plt.savefig('null_ratios_barplot.png')
plt.close()

# 判断如果空值比例 > 0.5，不保存该列
filtered_columns = [col for col in required_columns if null_ratios[col] <= 0.5]

# 第二步：筛选 case_number 包含 "沪" 的数据，即上海样本
df_shanghai = df[df['case_number'].str.contains('沪', na=False)].copy()

# 第三步：统计 incident_time 为空的比例
incident_time_null_ratio = df_shanghai['incident_time'].isnull().mean()
print(f"\nShanghai incident_time null ratio: {incident_time_null_ratio:.2%}")

# 第四步：根据比例规则处理
if incident_time_null_ratio > 0.05:
    # 如果 incident_time 为空，而 judgment_date 也是空，则删除
    df_shanghai = df_shanghai[~(
        df_shanghai['incident_time'].isnull() &
        df_shanghai['judgment_date'].isnull()
    )].copy()

    # 如果 incident_time 为空，使用 judgment_date 填充
    df_shanghai.loc[df_shanghai['incident_time'].isnull(), 'incident_time'] = \
        df_shanghai.loc[df_shanghai['incident_time'].isnull(), 'judgment_date']

# 第五步：保留空值比例低于 0.5 的字段
df_shanghai_cleaned = df_shanghai[filtered_columns].copy()

# 结果显示
print("\nCleaned Shanghai Dataset Preview:")
print(df_shanghai_cleaned.head())

# 保存
df_shanghai_cleaned.to_csv('ShanghaiCrimeCleaned01.csv', index=False)
