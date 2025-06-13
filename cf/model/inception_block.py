import torch
import torch.nn as nn


# 两种多尺度卷积特征提取器
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, dropout=0.5, use_residual=True, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.use_residual = use_residual
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        kernels = []
        bns = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            )
            bns.append(nn.BatchNorm2d(out_channels))
        self.kernels = nn.ModuleList(kernels)
        self.bns = nn.ModuleList(bns)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.spatial_dropout = nn.Dropout2d(0.5)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            out = self.kernels[i](x)
            out = self.bns[i](out)
            out = self.relu(out)
            out = self.norm(out)
            res_list.append(out)
        res = torch.stack(res_list, dim=-1).mean(-1)
        res = self.dropout(res)
        res = self.spatial_dropout(res)
        if self.use_residual and x.shape == res.shape:
            res = res + x
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[1, 2 * i + 3],
                    padding=[0, i + 1],
                )
            )
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[2 * i + 3, 1],
                    padding=[i + 1, 0],
                )
            )
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.spatial_dropout = nn.Dropout2d(0.5)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        res = self.norm(res)
        res = self.spatial_dropout(res)
        return res
