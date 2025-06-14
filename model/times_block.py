import torch
import torch.nn as nn


# 周期建模
class TimesBlock(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1
        )
        self.activation = nn.GELU()
        self.seq_len = seq_len

    def forward(self, x):  # x: [B, T, N, D]
        B, T, N, D = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N, D, T)  # [B*N, D, T]
        x_freq = torch.fft.rfft(x, dim=-1)
        x_out = self.conv(x_freq.real)  # [B*N, D, T//2+1]
        x = torch.fft.irfft(x_out, n=self.seq_len, dim=-1)
        x = self.activation(x).reshape(B, N, D, T).permute(0, 3, 1, 2)
        return x  # [B, T, N, D]
