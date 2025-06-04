# import torch
# import torch.nn as nn
# from model.dcgru_torch import DCGRU
# from model.times_block import TimesBlock
# from model.inception_block import Inception_Block_V1

# class GraphTimesNet(nn.Module):
#     def __init__(
#         self, supports, input_dim, hidden_dim, seq_len, horizon, use_inception=True
#     ):
#         super(GraphTimesNet, self).__init__()
#         self.seq_len = seq_len
#         self.horizon = horizon
#         self.num_nodes = supports[0].shape[0]

#         self.dcgru = DCGRU(
#             num_units=hidden_dim,
#             adj_mx=supports,
#             max_diffusion_step=2,
#             num_nodes=self.num_nodes,
#             input_dim=input_dim,
#             output_dim=hidden_dim, 
#             num_layers=1,
#             filter_type="laplacian"
#         )

#         self.use_inception = use_inception
#         if use_inception:
#             self.inception = Inception_Block_V1(
#                 in_channels=hidden_dim, out_channels=hidden_dim, num_kernels=6
#             )
        
#         self.times_block = TimesBlock(hidden_dim, seq_len)
#         self.output_proj = nn.Linear(hidden_dim, 8)

#     def forward(self, x):  # x: [B, T, N, D]
#         batch_size = x.size(0)
        
#         # Reshape for DCGRU: [B, T, N*D] -> [T, B, N*D]
#         x = x.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)
        
#         # Process sequence with DCGRU
#         outputs, _ = self.dcgru(x)  # outputs: [T, B, N*hidden_dim]
        
#         # Reshape back: [T, B, N*hidden_dim] -> [B, T, N, hidden_dim]
#         x = outputs.permute(1, 0, 2).reshape(batch_size, self.seq_len, self.num_nodes, -1)

#         if self.use_inception:
#             x = x.permute(0, 3, 1, 2)  # [B, D, T, N]
#             x = self.inception(x)
#             x = x.permute(0, 2, 3, 1)  # [B, T, N, D]

#         x = self.times_block(x)
#         x = self.output_proj(x)
#         return x[:, -self.horizon:, :, :]  # [B, horizon, N, 8]
    


import torch
import torch.nn as nn
from model.dcgru_torch import DCGRU
from model.times_block import TimesBlock
from model.inception_block import Inception_Block_V1

class GraphTimesNet(nn.Module):
    def __init__(
        self, supports, input_dim, hidden_dim, seq_len, horizon, use_inception=True
    ):
        super(GraphTimesNet, self).__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_nodes = supports[0].shape[0]
        self.hidden_dim = hidden_dim

        self.dcgru = DCGRU(
            num_units=hidden_dim,
            adj_mx=supports,
            max_diffusion_step=2,
            num_nodes=self.num_nodes,
            input_dim=input_dim,
            output_dim=hidden_dim, 
            num_layers=1,
            filter_type="laplacian"
        )

        self.use_inception = use_inception
        if use_inception:
            self.inception = Inception_Block_V1(
                in_channels=input_dim, out_channels=hidden_dim, num_kernels=6
            )
        
        # self.times_block = TimesBlock(hidden_dim, seq_len)

        self.times_block = TimesBlock(
            d_model=hidden_dim, 
            seq_len=seq_len, 
            num_nodes=self.num_nodes, 
            feature_dim=hidden_dim
        )
        self.output_proj = nn.Linear(hidden_dim, 8)

    def forward(self, x):  # x: [B, T, N, D]
        batch_size = x.size(0)
        
        # Reshape for DCGRU: [B, T, N*D] -> [T, B, N*D]
        x = x.reshape(batch_size, self.seq_len, -1).permute(1, 0, 2)
        
        # Process sequence with DCGRU
        outputs, _ = self.dcgru(x)  # outputs: [T, B, N*hidden_dim]
        
        # Reshape back: [T, B, N*hidden_dim] -> [B, T, N, hidden_dim]
        x = outputs.permute(1, 0, 2).reshape(batch_size, self.seq_len, self.num_nodes, -1)

        if self.use_inception:
            x = x.permute(0, 3, 1, 2)  # [B, D, T, N]
            x = self.inception(x)
            x = x.permute(0, 2, 3, 1)  # [B, T, N, D]


        x = self.times_block(x)
        x = self.output_proj(x)
        return x[:, -self.horizon:, :, :]  # [B, horizon, N, 8]