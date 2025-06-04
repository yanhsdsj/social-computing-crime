import torch
import torch.nn as nn


# # 两种多尺度卷积特征提取器
# class Inception_Block_V1(nn.Module):
#     def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
#         super(Inception_Block_V1, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_kernels = num_kernels
#         kernels = []
#         for i in range(self.num_kernels):
#             kernels.append(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
#             )
#         self.kernels = nn.ModuleList(kernels)
#         if init_weight:
#             self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         res_list = []
#         for i in range(self.num_kernels):
#             res_list.append(self.kernels[i](x))
#         res = torch.stack(res_list, dim=-1).mean(-1)
#         return res


# 可增加更多的卷积核大小或引入更深的卷积层
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            )
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for kernel in self.kernels:
            res_list.append(kernel(x))
        return torch.stack(res_list, dim=-1).mean(-1)
    


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
        return res



# import torch
# import torch.nn as nn

# class Inception_Block_V1(nn.Module):
#     def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_kernels = num_kernels
        
#         # 多尺度卷积核
#         kernels = []
#         for i in range(self.num_kernels):
#             kernels.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels, 
#                              kernel_size=2*i+1, padding=i),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU()
#                 )
#             )
#         self.kernels = nn.ModuleList(kernels)
        
#         if init_weight:
#             self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         res_list = []
#         for i in range(self.num_kernels):
#             res_list.append(self.kernels[i](x))
#         return torch.stack(res_list, dim=-1).mean(-1)

# class Inception_Block_V2(nn.Module):
#     def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_kernels = num_kernels
        
#         kernels = []
#         # 水平核
#         for i in range(num_kernels//2):
#             kernels.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels, 
#                              kernel_size=[1, 2*i+3], padding=[0, i+1]),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU()
#                 )
#             )
#             # 垂直核
#             kernels.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels,
#                              kernel_size=[2*i+3, 1], padding=[i+1, 0]),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU()
#                 )
#             )
#         # 1x1核
#         kernels.append(
#             nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU()
#             )
#         )
#         self.kernels = nn.ModuleList(kernels)
        
#         if init_weight:
#             self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         res_list = []
#         for kernel in self.kernels:
#             res_list.append(kernel(x))
#         return torch.stack(res_list, dim=-1).mean(-1)