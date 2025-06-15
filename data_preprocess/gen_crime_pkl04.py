import pandas as pd
import pickle
from datetime import datetime


def csv_to_pkl(csv_path, output_pkl_path):
    # 1. 读取犯罪数据
    df = pd.read_csv(csv_path)

    # --------------------- 安全字段检测 ---------------------
    expected_cols = ['incident_time', 'latitude', 'longitude', 'crime_type', 'crime_type_id', 'district_id']
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"列缺失：{col}，请确认 {csv_path} 文件结构正确")

    # 2. 时间字段处理
    df['date_occ'] = pd.to_datetime(df['incident_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['combine_time'] = df['date_occ'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%MZ') if pd.notnull(x) else None)

    # 3. 社区字段标准化
    df['neighborhood_id'] = df['district_id'].astype(int)

    # 4. 犯罪类型英文名创建（使用映射）
    # 定义静态映射（可从你 CSV 或外部文件中导入）
    crime_mapping = {
        "盗窃": "theft",
        "抢劫": "robbery",
        "诈骗": "fraud",
        "故意伤害": "aggravated_assault",
        "寻衅滋事": "disorderly_conduct",
        "贩卖毒品": "drug_trafficking",
        "强奸": "rape",
        "危险驾驶": "dangerous_driving"
    }

    df['new_category'] = df['crime_type'].map(crime_mapping).fillna("other")

    # 5. 生成新的 crime_type_id（英文分类 → 数字ID）
    crime_types = df['new_category'].unique()
    crime_id_mapping = {crime: idx for idx, crime in enumerate(crime_types)}
    df['crime_type_id'] = df['new_category'].map(crime_id_mapping)

    # 6. 最终字段过滤和缺失处理
    required_columns = ['neighborhood_id', 'new_category', 'crime_type_id', 'date_occ',
                        'combine_time', 'latitude', 'longitude']
    df = df[required_columns].dropna()

    # 7. 保存
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(df, f, protocol=2)

    print(f"✅ 成功保存为 {output_pkl_path}, 样本数: {len(df)}")
    print(f"包含字段: {list(df.columns)}")


# 示例调用（替换为实际文件路径）
csv_to_pkl(
    csv_path='data/ShanghaiCrimeMapped03.csv',
    output_pkl_path='data/ShanghaiCrimeData04.pkl'
)