import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return torch.sigmoid(out).view(x.size(0), x.size(1), 1, 1)


# 两种多尺度卷积特征提取器
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, dropout=0.3, use_residual=True, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.use_residual = use_residual
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        # 添加通道注意力机制
        self.channel_attention = ChannelAttention(out_channels)
        
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
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
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
        
        # 应用通道注意力
        attention = self.channel_attention(res)
        res = res * attention
        
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
        
        # 添加通道注意力机制
        self.channel_attention = ChannelAttention(out_channels)
        
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
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            out = self.kernels[i](x)
            out = self.relu(out)
            res_list.append(out)
        res = torch.stack(res_list, dim=-1).mean(-1)
        
        # 应用通道注意力
        attention = self.channel_attention(res)
        res = res * attention
        
        res = self.norm(res)
        res = self.spatial_dropout(res)
        return res
