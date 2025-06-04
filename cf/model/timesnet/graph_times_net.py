import torch
import torch.nn as nn
from model.graph_conv import GraphConv
from model.times_block import TimesBlock
from model.inception_block import Inception_Block_V1


class GraphTimesNet(nn.Module):
    def __init__(
        self, supports, input_dim, hidden_dim, seq_len, horizon, use_inception=True
    ):
        super(GraphTimesNet, self).__init__()
        # 图卷积：处理邻接节点的空间关系
        self.graph_conv = GraphConv(supports, input_dim, hidden_dim)
        self.use_inception = use_inception
        if use_inception:
            # 多尺度卷积块：提取时空交互特征
            self.inception = Inception_Block_V1(
                in_channels=hidden_dim, out_channels=hidden_dim, num_kernels=6
            )
        # 频域建模模块：提取每个节点在时间维度的周期性
        self.times_block = TimesBlock(hidden_dim, seq_len)
        # 线性层：映射成预测目标维度
        self.output_proj = nn.Linear(hidden_dim, 8)
        self.horizon = horizon

    # 前向传播逻辑
    def forward(self, x):  # x: [B, T, N, D]
        x = self.graph_conv(x)  # [B, T, N, hidden]
        if self.use_inception:
            x = x.permute(0, 3, 1, 2)  # [B, D, T, N]
            x = self.inception(x)  # Inception 输出: [B, D, T, N]
            x = x.permute(0, 2, 3, 1)  # → [B, T, N, D]
        x = self.times_block(x)  # [B, T, N, hidden]
        x = self.output_proj(x)  # [B, T, N, out_dim]
        return x[:, -self.horizon :, :, :]  # 取最后 horizon 步预测
