import pickle
import numpy as np
import scipy.sparse as sp
import torch


# 从.pkl文件加载图结构
def load_graph_data(pkl_filename):
    with open(pkl_filename, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")
    return sensor_ids, sensor_id_to_ind, adj_mx


# 计算归一化的拉普拉斯矩阵
def calculate_scaled_laplacian(adj_mx, lambda_max=2):
    adj = sp.coo_matrix(adj_mx)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(
        d_mat_inv_sqrt
    )
    if lambda_max is None:
        lambda_max = sp.linalg.eigs(laplacian, k=1, which="LR")[0].real
    laplacian = (2 / lambda_max * laplacian) - sp.eye(adj.shape[0])
    return laplacian.astype(np.float32)


# 计算随机游走归一化矩阵D-1A
def calculate_random_walk_matrix(adj_mx):
    adj = sp.coo_matrix(adj_mx)
    d = np.array(adj.sum(1)).flatten()
    d_inv = np.power(d, -1)
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj).astype(np.float32)


# 将scipy稀疏矩阵转为PyTorch的稀疏张量
def build_sparse_tensor(matrix):
    matrix = sp.coo_matrix(matrix)
    indices = torch.tensor(np.vstack((matrix.row, matrix.col)).T, dtype=torch.long)
    values = torch.tensor(matrix.data, dtype=torch.float32)
    shape = torch.Size(matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
