# 导入 gen_adj_mx 中的核心函数和依赖库
import pandas as pd
import pickle

from gen_adj_mx import get_adjacency_matrix


def generate_adjacency_matrix(sensor_ids_path, distances_path, output_pkl, normalized_k=0.1):
    """
    直接调用 gen_adj_mx 逻辑生成邻接矩阵

    :param sensor_ids_path: 邻域ID文件路径（如 shanghai_neighborhood_ids.txt）
    :param distances_path: 距离文件路径（如 shanghai_neighborhood_distances.csv）
    :param output_pkl: 输出邻接矩阵的 pkl 文件路径（如 adj_mx_shanghai.pkl）
    :param normalized_k: 归一化阈值，默认 0.1
    """
    # 1. 读取邻域ID列表（从 txt 文件中加载）
    with open(sensor_ids_path, 'r') as f:
        sensor_ids = f.read().strip().split(',')  # 格式：['1', '2', '3', ...]

    # 2. 加载距离数据（确保 'from' 和 'to' 列与 sensor_ids 中的 ID 一致）
    distance_df = pd.read_csv(distances_path, dtype={'from': str, 'to': str})  # 保持 ID 为字符串类型

    # 3. 生成邻接矩阵
    sensor_ids, sensor_id_to_ind, adj_mx = get_adjacency_matrix(
        distance_df=distance_df,
        sensor_ids=sensor_ids,
        normalized_k=normalized_k
    )

    # 4. 保存结果到 pkl 文件
    with open(output_pkl, 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)

    print(f"邻接矩阵已成功生成并保存至 {output_pkl}")


def print_adjacency_matrix_info(pkl_path):
    """
    打印邻接矩阵文件中的信息

    :param pkl_path: 邻接矩阵的 pkl 文件路径（如 adj_mx_shanghai.pkl）
    """
    # 1. 加载 pkl 文件中的内容
    with open(pkl_path, 'rb') as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f)

    # 2. 打印传感器ID列表信息
    print("传感器ID列表（sensor_ids）:")
    print(f"类型: {type(sensor_ids)}")
    print(f"长度: {len(sensor_ids)}")
    print(f"前10个ID示例: {sensor_ids[:10]}")
    print()

    # 3. 打印传感器ID到索引的映射信息
    print("传感器ID到索引的映射（sensor_id_to_ind）:")
    print(f"类型: {type(sensor_id_to_ind)}")
    print(f"字典项数: {len(sensor_id_to_ind)}")
    print(f"前10个映射示例: {list(sensor_id_to_ind.items())[:10]}")
    print()

    # 4. 打印邻接矩阵信息
    print("邻接矩阵（adj_mx）:")
    print(f"类型: {type(adj_mx)}")
    print(f"形状: {adj_mx.shape}")
    print(f"矩阵前5行和前5列示例:\n{adj_mx[:5, :5]}")
    print()

# generate_adjacency_matrix(
#     sensor_ids_path='data/distinct.txt',
#     distances_path='data/shanghai_distinct_distances06.csv',
#     output_pkl='data/adj_mx_shanghai07.pkl',
#     normalized_k=0.1
# )

print_adjacency_matrix_info('data/adj_mx_shanghai07.pkl')