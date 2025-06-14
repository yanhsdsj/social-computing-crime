import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, supports, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.supports = supports  # list of [N, N]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x):  # x: [B, T, N, input_dim]
        B, T, N, D = x.shape
        assert (
            D == self.input_dim
        ), f"Input feature dim mismatch: {D} vs expected {self.input_dim}"

        # [B, T, N, D] → [B*T, N, D]
        x = x.reshape(-1, N, D)

        # 图卷积：每个 support 矩阵做传播
        out = 0
        for support in self.supports:
            support = support.to(x.device)  # [N, N]
            # 使用enisum实现图卷积的聚合操作 对每个batch和时间不，对节点的特征进行加权聚合
            support_x = torch.einsum("ij,bjd->bid", support, x)  # [B*T, N, D]
            # 线性层，获取想要的维度
            out += torch.matmul(support_x, self.weights)  # [B*T, N, output_dim]

        out = out + self.bias
        out = out.reshape(B, T, N, self.output_dim)
        return out
