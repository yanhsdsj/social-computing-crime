import torch
import torch.nn as nn


# # 周期建模
# class TimesBlock(nn.Module):
#     def __init__(self, d_model, seq_len):
#         super().__init__()
#         self.conv = nn.Conv1d(
#             in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1
#         )
#         self.activation = nn.GELU()
#         self.seq_len = seq_len

#     def forward(self, x):  # x: [B, T, N, D]
#         B, T, N, D = x.shape
#         x = x.permute(0, 2, 3, 1).reshape(B * N, D, T)  # [B*N, D, T]
#         x_freq = torch.fft.rfft(x, dim=-1)
#         x_out = self.conv(x_freq.real)  # [B*N, D, T//2+1]
#         x = torch.fft.irfft(x_out, n=self.seq_len, dim=-1)
#         x = self.activation(x).reshape(B, N, D, T).permute(0, 3, 1, 2)
#         return x  # [B, T, N, D]


# # 在 TimesBlock 中引入残差连接
# class TimesBlock(nn.Module):
#     def __init__(self, d_model, seq_len, expansion=2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.d_model = d_model
        
#         # 频域处理
#         self.fft_conv = nn.Sequential(
#             nn.Conv1d(d_model, d_model * expansion, 3, padding=1),
#             nn.BatchNorm1d(d_model * expansion),
#             nn.GELU(),
#             nn.Conv1d(d_model * expansion, d_model, 3, padding=1),
#             nn.BatchNorm1d(d_model)
#         )
        
#         # 时域处理
#         self.time_conv = nn.Sequential(
#             nn.Conv1d(d_model, d_model, 3, padding=1),
#             nn.BatchNorm1d(d_model),
#             nn.GELU()
#         )
        
#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, x):  # x: [B, T, N, D]
#         B, T, N, D = x.shape
#         residual = x
        
#         # 转换为频域 [B*N, D, T]
#         x = x.permute(0, 2, 3, 1).reshape(B * N, D, T)
#         x_freq = torch.fft.rfft(x, dim=-1)
        
#         # 频域卷积
#         freq_out = self.fft_conv(x_freq.real)
#         x = torch.fft.irfft(freq_out, n=self.seq_len, dim=-1)
        
#         # 时域卷积
#         x = self.time_conv(x)
        
#         # 恢复形状并添加残差
#         x = x.reshape(B, N, D, T).permute(0, 3, 1, 2)
#         x = self.norm(x + residual)
        
#         return x  # [B, T, N, D]


# 引入注意力机制
class TimesBlock(nn.Module):
    def __init__(self, d_model, seq_len, num_nodes, feature_dim, expansion=2):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        
        # 计算嵌入维度
        self.embed_dim = num_nodes * feature_dim
        
        # 频域处理
        self.fft_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * expansion, 3, padding=1),
            nn.BatchNorm1d(d_model * expansion),
            nn.GELU(),
            nn.Conv1d(d_model * expansion, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model)
        )
        
        # 时域处理
        self.time_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, batch_first=True)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # x: [B, T, N, D]
        B, T, N, D = x.shape
        residual = x
        
        # 转换为频域 [B*N, D, T]
        x = x.permute(0, 2, 3, 1).reshape(B * N, D, T)
        x_freq = torch.fft.rfft(x, dim=-1)
        
        # 频域卷积
        freq_out = self.fft_conv(x_freq.real)
        x = torch.fft.irfft(freq_out, n=self.seq_len, dim=-1)
        
        # 时域卷积
        x = self.time_conv(x)
        
        # 恢复形状并添加残差
        x = x.reshape(B, N, D, T).permute(0, 3, 1, 2)
        
        # 注意力机制
        # 调整输入形状为 [B, T, N * D]
        x = x.reshape(B, T, -1)  # [B, T, N * D]
        x, _ = self.attention(x, x, x)
        x = x.reshape(B, T, N, D)  # 恢复形状为 [B, T, N, D]
        
        x = self.norm(x + residual)
        
        return x  # [B, T, N, D]