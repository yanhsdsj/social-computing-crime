import os
import pickle
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from zoneinfo import ZoneInfo
from sklearn.model_selection import train_test_split

# ==================== 基础配置 ====================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
OUTPUT_DIR = "data/CRIME-SHANGHAI/8"  # 输出目录
# ==================== 时间窗口偏移配置 ====================
WINDOW_SIZE = 10  # 增加窗口大小
HISTORY_STEPS = WINDOW_SIZE - 1  # 输入特征的时间步数（默认窗口大小减1用于特征，1用于标签）
PREDICT_STEPS = 1                # 预测未来时间步数（默认预测1步）

# 定义x_offsets和y_offsets（相对当前时间的偏移）
x_offsets = np.arange(-HISTORY_STEPS + 1, 1, 1).reshape(-1, 1)  # 形状 (7, 1)
y_offsets = np.arange(1, PREDICT_STEPS + 1, 1).reshape(-1, 1)  # 形状 (1, 1)

# ==================== 加载数据 ====================
print("\n===== 正在加载数据 =====")
def preprocess_time(crime_data):
    # 1. 将时间字符串转换为无时区的 datetime 对象
    crime_data["combine_time"] = pd.to_datetime(crime_data["combine_time"])

    # 2. 检查是否已有时区，再决定是添加还是转换时区
    if crime_data["combine_time"].dt.tz is None:
        # 无 timezone，添加上海时区
        crime_data["combine_time"] = crime_data["combine_time"].dt.tz_localize("Asia/Shanghai")
    else:
        # 已有 timezone，转换为上海时区（如果需要）
        crime_data["combine_time"] = crime_data["combine_time"].dt.tz_convert("Asia/Shanghai")

    # 验证时区（应输出 "Asia/Shanghai"）
    print("combine_time 时区:", crime_data["combine_time"].dt.tz)
    return crime_data

# 加载数据并预处理时间
shanghai_crime = pd.read_pickle('data/ShanghaiCrimeData04.pkl')
shanghai_crime = preprocess_time(shanghai_crime)
print(f"数据加载完成！样本量: {len(shanghai_crime)} 条")
print(f"关键字段检查: {shanghai_crime.columns.tolist()}")
print(f"社区数量（neighborhood_id唯一值）: {shanghai_crime['neighborhood_id'].nunique()}")
print(f"犯罪类型数量（crime_type_id唯一值）: {shanghai_crime['crime_type_id'].nunique()}")

# ==================== 时间槽生成 ====================
print("\n===== 正在生成时间槽 =====")

def convert(date_time):
    data_format = '%m/%d/%Y %H:%M:%S'  # 原始时间格式
    try:
        return datetime.strptime(date_time, data_format)
    except ValueError:
        return pd.NaT  # 无效时间返回NaT（Not a Time）

def datetime_range(start, end, delta):
    """生成时间槽列表"""
    current = start
    while current < end:
        yield current
        current += delta

def is_valid_time_format(time_str, target_format='%Y-%m-%dT%H:%MZ'):
    """
    验证时间字符串是否符合指定格式
    :param time_str: 待验证的时间字符串
    :param target_format: 目标格式（默认 '%Y-%m-%dT%H:%MZ'）
    :return: 符合格式返回 True，否则返回 False
    """
    try:
        datetime.strptime(time_str, target_format)
        return True
    except ValueError:
        return False

# 提取全局时间范围
start_time = shanghai_crime["combine_time"].min()  # 已本地化为 Asia/Shanghai
end_time = shanghai_crime["combine_time"].max()
print(f"时间范围类型: start={type(start_time)}, end={type(end_time)}")
print(f"时间范围: {start_time.strftime('%Y-%m-%d %H:%M')} 至 {end_time.strftime('%Y-%m-%d %H:%M')}")

# 生成时间槽（每天一个时间片）
time_frequency = timedelta(days=1)
time_slots = [dt.strftime('%Y-%m-%dT%H:%MZ') for dt in
              datetime_range(start_time, end_time, time_frequency)]
print(f"生成时间槽数量: {len(time_slots)} 个（频率: 每天）")
print("前5个时间槽:", time_slots[:5])  # 应输出类似['1989-10-18T15:00Z', '1989-10-19T15:00Z', ...]
print("后5个时间槽:", time_slots[-5:])  # 应接近2019-12-17的时间

# 检查所有时间槽的格式
invalid_slots = [slot for slot in time_slots if not is_valid_time_format(slot)]

if invalid_slots:
    print(f"发现 {len(invalid_slots)} 个格式错误的时间槽:")
    for idx, slot in enumerate(invalid_slots[:5]):  # 仅打印前5个错误示例
        print(f"  {idx+1}. {slot}")
    raise ValueError("时间槽格式不符合 '%Y-%m-%dT%H:%MZ'，请检查时间生成逻辑！")
else:
    print("所有时间槽格式验证通过 ✔")

# ==================== 关键参数计算 ====================
num_nodes = shanghai_crime["neighborhood_id"].nunique()  # 社区总数
num_crime_types = shanghai_crime["crime_type_id"].nunique()  # 犯罪类型总数
print(f"\n===== 关键参数 =====")
print(f"社区数量 (num_nodes): {num_nodes}")
print(f"犯罪类型数量 (num_crime_types): {num_crime_types}")


# ==================== 滑动窗口生成 ====================
def moving_window(x, window_size):
    # 确保窗口大小不超过时间槽长度
    if window_size > len(x):
        raise ValueError("窗口大小超过时间槽数量")
    # 生成连续的滑动窗口（步长1）
    return [x[i:i+window_size] for i in range(len(x) - window_size + 1)]

# 生成窗口后验证
time_windows = moving_window(time_slots, WINDOW_SIZE)
print("第一个窗口:", time_windows[0])  # 应包含8个连续时间字符串
print("最后一个窗口:", time_windows[-1])  # 应包含最后8个时间字符串

# ==================== 特征与标签构建 ====================
print("\n===== 正在构建特征与标签（耗时操作） =====")


def build_features_labels(crime_data, time_slots, window_size, num_nodes, num_crime_types):
    features = []
    labels = []
    total_windows = len(time_slots) - window_size

    for i in tqdm(range(total_windows), desc="处理窗口", unit="窗口"):
        feature_windows = time_slots[i:i + window_size - 1]
        feature = []
        for slot in feature_windows:
            # 解析时间槽字符串并本地化为上海时区（关键）
            start = datetime.strptime(slot, '%Y-%m-%dT%H:%MZ').replace(tzinfo=ZoneInfo("Asia/Shanghai"))
            end = start + timedelta(days=1)

            # 过滤上海时间范围内的数据（时区一致，可直接比较）
            df_window = crime_data[
                (crime_data["combine_time"] >= start) &
                (crime_data["combine_time"] < end)
                ]

            # 统计犯罪次数（逻辑不变）
            count_matrix = np.zeros((num_nodes, num_crime_types), dtype=np.int8)
            for _, row in df_window.iterrows():
                neigh_id = int(row["neighborhood_id"])
                crime_id = int(row["crime_type_id"])
                if 0 <= neigh_id < num_nodes and 0 <= crime_id < num_crime_types:
                    count_matrix[neigh_id, crime_id] += 1
            feature.append(count_matrix)
        features.append(feature)

        # 标签部分同理（使用上海时区）
        label_slot = time_slots[i + window_size - 1]
        label_start = datetime.strptime(label_slot, '%Y-%m-%dT%H:%MZ').replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        label_end = label_start + timedelta(days=1)
        df_label = crime_data[
            (crime_data["combine_time"] >= label_start) &
            (crime_data["combine_time"] < label_end)
            ]
        label_counts = np.zeros((num_nodes, num_crime_types), dtype=np.int8)
        for _, row in df_label.iterrows():
            neigh_id = int(row["neighborhood_id"])
            crime_id = int(row["crime_type_id"])
            if 0 <= neigh_id < num_nodes and 0 <= crime_id < num_crime_types:
                label_counts[neigh_id, crime_id] = 1
        labels.append(np.expand_dims(label_counts, axis=0))

    return np.array(features), np.array(labels)


# 执行特征标签构建（带进度条）
x, y = build_features_labels(shanghai_crime, time_slots, WINDOW_SIZE, num_nodes, num_crime_types)
print(f"\n特征形状: {x.shape}（样本数, 时间槽数, 社区数, 犯罪类型数）")
print(f"标签形状: {y.shape}（样本数, 社区数, 犯罪类型数）")

# ==================== 数据集划分 ====================
print("\n===== 正在划分训练/验证/测试集 =====")


def split_dataset(x, y, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # 首先将数据划分为训练集和临时集（验证集 + 测试集）
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, train_size=train_ratio, random_state=42)
    # 计算验证集在临时集中的比例
    val_ratio_temp = val_ratio / (val_ratio + test_ratio)
    # 从临时集中划分出验证集和测试集
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, train_size=val_ratio_temp, random_state=42)

    print(f"总样本数: {x.shape[0]}")
    print(f"训练集: {x_train.shape[0]} 样本（{train_ratio * 100:.1f}%）")
    print(f"验证集: {x_val.shape[0]} 样本（{val_ratio * 100:.1f}%）")
    print(f"测试集: {x_test.shape[0]} 样本（{test_ratio * 100:.1f}%）")
    return x_train, y_train, x_val, y_val, x_test, y_test

x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(x, y, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

# 数据增强
def data_augmentation(x_train, y_train, multiplier=1.2):
    num_samples = x_train.shape[0]
    num_new_samples = int(num_samples * (multiplier - 1))

    # 随机选择样本进行复制
    indices = np.random.choice(num_samples, num_new_samples)
    x_augmented = np.vstack((x_train, x_train[indices]))
    y_augmented = np.vstack((y_train, y_train[indices]))

    return x_augmented, y_augmented

x_train, y_train = data_augmentation(x_train, y_train)

# 检查非零元素比例
def non_zero_ratio(arr):
    return np.count_nonzero(arr) / arr.size

print(f"训练集x非零元素比例: {non_zero_ratio(x_train):.4f}")
print(f"训练集y非零元素比例: {non_zero_ratio(y_train):.4f}")
print(f"验证集x非零元素比例: {non_zero_ratio(x_val):.4f}")
print(f"验证集y非零元素比例: {non_zero_ratio(y_val):.4f}")
print(f"测试集x非零元素比例: {non_zero_ratio(x_test):.4f}")
print(f"测试集y非零元素比例: {non_zero_ratio(x_test):.4f}")

# ==================== 保存数据集 ====================
print("\n===== 正在保存数据集 =====")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 保存训练集
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "train.npz"),
    x=x_train,
    y=y_train,
    x_offsets=x_offsets,
    y_offsets=y_offsets
)
# 保存验证集
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "val.npz"),
    x=x_val,
    y=y_val,
    x_offsets=x_offsets,
    y_offsets=y_offsets
)
# 保存测试集
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "test.npz"),
    x=x_test,
    y=y_test,
    x_offsets=x_offsets,
    y_offsets=y_offsets
)


# 验证保存文件大小
def get_file_size(file_path):
    return f"{os.path.getsize(file_path) / 1024 / 1024:.2f} MB"


print(
    f"训练集保存路径: {os.path.join(OUTPUT_DIR, 'train.npz')}, 大小: {get_file_size(os.path.join(OUTPUT_DIR, 'train.npz'))}")
print(
    f"验证集保存路径: {os.path.join(OUTPUT_DIR, 'val.npz')}, 大小: {get_file_size(os.path.join(OUTPUT_DIR, 'val.npz'))}")
print(
    f"测试集保存路径: {os.path.join(OUTPUT_DIR, 'test.npz')}, 大小: {get_file_size(os.path.join(OUTPUT_DIR, 'test.npz'))}")

# ==================== 最终验证 ====================
print("\n===== 数据验证 =====")
train_data = np.load(os.path.join(OUTPUT_DIR, "train.npz"))
x_train_loaded = train_data["x"]
y_train_loaded = train_data["y"]
print(f"加载的训练集x形状: {x_train_loaded.shape}（与原始一致: {np.array_equal(x_train, x_train_loaded)}）")
print(f"加载的训练集y形状: {y_train_loaded.shape}（与原始一致: {np.array_equal(y_train, y_train_loaded)}）")

print("\n===== 全流程完成！ =====")