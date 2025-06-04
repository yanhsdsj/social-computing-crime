import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


# class DCGRUCell(nn.Module):
#     """Graph Convolution Gated Recurrent Unit cell - PyTorch implementation.
#     """
#     def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, input_dim, num_proj=None,
#              activation=torch.tanh, filter_type="laplacian", use_gc_for_ru=True):
#         super(DCGRUCell, self).__init__()
#         self._activation = activation
#         self._num_nodes = num_nodes
#         self._num_proj = num_proj
#         self._num_units = num_units
#         self._max_diffusion_step = max_diffusion_step
#         self._use_gc_for_ru = use_gc_for_ru

#         # Build support matrices
#         self._supports = []
#         supports = []

#         if filter_type == "laplacian":
#             supports.append(self._calculate_scaled_laplacian(adj_mx))
#         elif filter_type == "random_walk":
#             supports.append(self._calculate_random_walk_matrix(adj_mx).T)
#         elif filter_type == "dual_random_walk":
#             supports.append(self._calculate_random_walk_matrix(adj_mx).T)
#             supports.append(self._calculate_random_walk_matrix(adj_mx.T).T)
#         else:
#             supports.append(self._calculate_scaled_laplacian(adj_mx))

#         # Convert to sparse tensors
#         for support in supports:
#             self._supports.append(self._build_sparse_tensor(support))

#         # Calculate dimensions for weight matrices
#         num_matrices = len(self._supports) * self._max_diffusion_step + 1

#         # Initialize weight matrices
#         if self._use_gc_for_ru:
#             # Graph convolution weights for reset and update gates
#             self.gate_weights = nn.Parameter(
#                 torch.empty(num_matrices * (input_dim + num_units), 2 * num_units)
#             )
#             nn.init.xavier_uniform_(self.gate_weights)
#             self.gate_biases = nn.Parameter(torch.ones(2 * num_units))  # Bias of 1.0 for GRU
#         else:
#             # Fully connected weights for reset and update gates
#             self.fc_gate_weights = nn.Parameter(
#                 torch.empty(2 * num_units, 2 * num_units)
#             )
#             nn.init.xavier_uniform_(self.fc_gate_weights)
#             self.fc_gate_biases = nn.Parameter(torch.ones(2 * num_units))

#         # Graph convolution weights for candidate state
#         self.candidate_weights = nn.Parameter(
#             torch.empty(num_matrices * (input_dim + num_units), num_units)
#         )
#         nn.init.xavier_uniform_(self.candidate_weights)
#         self.candidate_biases = nn.Parameter(torch.zeros(num_units))

#         # Projection weights (optional)
#         if self._num_proj is not None:
#             self.proj_weights = nn.Parameter(torch.empty(num_units, num_proj))
#             nn.init.xavier_uniform_(self.proj_weights)

            
    
#     @staticmethod
#     def _calculate_scaled_laplacian(adj_mx, lambda_max=2.0):
#         """Calculate scaled Laplacian matrix."""
#         # adj_mx = sp.coo_matrix(adj_mx)
        
#         adj_tensor = adj_mx[0] if isinstance(adj_mx, list) else adj_mx
#         adj_mx = sp.coo_matrix(adj_tensor.cpu().numpy())


#         d = np.array(adj_mx.sum(1)).flatten()
#         d_inv_sqrt = np.power(d, -0.5)
#         d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#         d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#         normalized_laplacian = sp.eye(adj_mx.shape[0]) - d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt)
        
#         if lambda_max is None:
#             lambda_max = sp.linalg.eigsh(normalized_laplacian, 1, which='LA', return_eigenvectors=False)[0]
        
#         return (2 / lambda_max * normalized_laplacian - sp.eye(normalized_laplacian.shape[0])).astype(np.float32)
    
#     @staticmethod
#     def _calculate_random_walk_matrix(adj_mx):
#         """Calculate random walk matrix."""
#         adj_mx = sp.coo_matrix(adj_mx)
#         d = np.array(adj_mx.sum(1)).flatten()
#         d_inv = np.power(d, -1)
#         d_inv[np.isinf(d_inv)] = 0.
#         d_mat_inv = sp.diags(d_inv)
#         random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
#         return random_walk_mx.astype(np.float32)
    
#     def _build_sparse_tensor(self, sparse_mx):
#         """Convert scipy sparse matrix to PyTorch sparse tensor."""
#         sparse_mx = sparse_mx.tocoo().astype(np.float32)
#         indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#         values = torch.from_numpy(sparse_mx.data)
#         shape = torch.Size(sparse_mx.shape)
#         return torch.sparse_coo_tensor(indices, values, shape)
    
#     @property
#     def state_size(self):
#         return self._num_nodes * self._num_units
    
#     @property
#     def output_size(self):
#         output_size = self._num_nodes * self._num_units
#         if self._num_proj is not None:
#             output_size = self._num_nodes * self._num_proj
#         return output_size
    
#     def _gconv(self, inputs, state, output_size, bias_start=0.0, weights=None, biases=None):
#         """Graph convolution between input and the graph matrix."""
#         batch_size = inputs.size(0)
#         # print("weights shape:", weights.shape)  


#         # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
#         inputs = inputs.view(batch_size, self._num_nodes, -1)
#         # print("inputs shape:", inputs.shape)

#         state = state.view(batch_size, self._num_nodes, -1)
#         # print("state shape:", state.shape)
        
#         inputs_and_state = torch.cat([inputs, state], dim=2)
#         input_size = inputs_and_state.size(2)
#         # print("input_size:", input_size)
#         # print("inputs_and_state shape:", inputs_and_state.shape)

        
#         # Transpose for graph convolution: (num_nodes, input_size, batch_size)
#         x = inputs_and_state.permute(1, 2, 0)
#         x0 = x.contiguous().view(self._num_nodes, input_size * batch_size)
#         x_list = [x0]
        
#         # Apply diffusion steps
#         if self._max_diffusion_step > 0:
#             for support in self._supports:
#                 # x1 = torch.sparse.mm(support, x0)
#                 support = support.to(x0.device)
#                 x1 = torch.sparse.mm(support, x0)

#                 x_list.append(x1)
                
#                 for k in range(2, self._max_diffusion_step + 1):
#                     x2 = 2 * torch.sparse.mm(support, x1) - x0
#                     x_list.append(x2)
#                     x1, x0 = x2, x1
        
#         # Stack all diffusion results
#         num_matrices = len(x_list)
#         # print("num_matrices:", num_matrices)

#         x = torch.stack(x_list, dim=0)  # (num_matrices, num_nodes, input_size * batch_size)
#         x = x.view(num_matrices, self._num_nodes, input_size, batch_size)
#         x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, num_matrices)
#         x = x.contiguous().view(batch_size * self._num_nodes, input_size * num_matrices)
        
#         # Apply linear transformation
#         x = torch.matmul(x, weights) + biases
#         # print("x shape before matmul:", x.shape)
        
                
#         # Reshape back to (batch_size, num_nodes * output_size)
#         return x.view(batch_size, self._num_nodes * output_size)
    
#     def _fc(self, inputs, state, output_size):
#         """Fully connected layer for gates."""
#         batch_size = inputs.size(0)
#         inputs = inputs.view(batch_size * self._num_nodes, -1)
#         state = state.view(batch_size * self._num_nodes, -1)
#         inputs_and_state = torch.cat([inputs, state], dim=-1)
        
#         output = torch.matmul(inputs_and_state, self.fc_gate_weights) + self.fc_gate_biases
#         return output
    
#     def forward(self, inputs, state):
#         """
#         Forward pass of DCGRU cell.
        
#         Args:
#             inputs: Input tensor of shape (batch_size, num_nodes * input_dim)
#             state: Hidden state tensor of shape (batch_size, num_nodes * num_units)
            
#         Returns:
#             output: Output tensor
#             new_state: New hidden state
#         """
#         batch_size = inputs.size(0)
        
#         # Calculate gates (reset and update)
#         if self._use_gc_for_ru:
#             gate_inputs = self._gconv(inputs, state, 2 * self._num_units, 
#                                     bias_start=1.0, weights=self.gate_weights, biases=self.gate_biases)
#         else:
#             gate_inputs = self._fc(inputs, state, 2 * self._num_units)
        
#         gate_inputs = torch.sigmoid(gate_inputs)
#         gate_inputs = gate_inputs.view(batch_size, self._num_nodes, 2 * self._num_units)
#         r, u = torch.split(gate_inputs, self._num_units, dim=-1)
#         r = r.contiguous().view(batch_size, self._num_nodes * self._num_units)
#         u = u.contiguous().view(batch_size, self._num_nodes * self._num_units)
        
#         # Calculate candidate state
#         candidate = self._gconv(inputs, r * state, self._num_units, 
#                               weights=self.candidate_weights, biases=self.candidate_biases)
#         if self._activation is not None:
#             candidate = self._activation(candidate)
        
#         # Calculate new state
#         new_state = u * state + (1 - u) * candidate
#         output = new_state
        
#         # Apply projection if specified
#         if self._num_proj is not None:
#             output = output.view(batch_size * self._num_nodes, self._num_units)
#             output = torch.matmul(output, self.proj_weights)
#             output = output.view(batch_size, self.output_size)
        
#         return output, new_state


# class DCGRU(nn.Module):
#     """Multi-layer DCGRU model."""
    
#     def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes,
#                  input_dim, output_dim, num_layers=1, num_proj=None,
#                  activation=torch.tanh, filter_type="laplacian", use_gc_for_ru=True):
#         """
#         Args:
#             num_units: Hidden units per DCGRU layer
#             adj_mx: Adjacency matrix
#             max_diffusion_step: Maximum diffusion steps
#             num_nodes: Number of graph nodes
#             input_dim: Input feature dimension
#             output_dim: Output feature dimension
#             num_layers: Number of DCGRU layers
#             num_proj: Projection dimension (optional)
#             activation: Activation function
#             filter_type: Filter type for graph convolution
#             use_gc_for_ru: Use graph convolution for update/reset gates
#         """
#         super(DCGRU, self).__init__()
#         self.num_layers = num_layers
#         self.num_nodes = num_nodes
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         cells = []
#         for i in range(num_layers):
#             cell_input_dim = input_dim if i == 0 else num_units
#             cell = DCGRUCell(
#                 num_units=num_units,
#                 adj_mx=adj_mx,
#                 max_diffusion_step=max_diffusion_step,
#                 num_nodes=num_nodes,
#                 input_dim=cell_input_dim,
#                 num_proj=num_proj if i == num_layers - 1 else None,
#                 activation=activation,
#                 filter_type=filter_type,
#                 use_gc_for_ru=use_gc_for_ru
#             )
#             cells.append(cell)
#         self.cells = nn.ModuleList(cells)

#         # Optional final fully connected layer to map to output_dim
#         if num_proj is not None:
#             self.fc_out = nn.Linear(num_proj * num_nodes, output_dim * num_nodes)
#         else:
#             self.fc_out = nn.Linear(num_units * num_nodes, output_dim * num_nodes)

#     def forward(self, inputs, hidden_state=None):
#         """
#         Args:
#             inputs: Input tensor (batch_size, seq_len, num_nodes * input_dim)
#             hidden_state: Optional list of hidden states per layer

#         Returns:
#             outputs: Tensor of shape (batch_size, seq_len, num_nodes * output_dim)
#             final_state: Final hidden states
#         """
#         batch_size, seq_len, _ = inputs.shape
#         if hidden_state is None:
#             hidden_state = [
#                 torch.zeros(batch_size, self.cells[i].state_size, device=inputs.device)
#                 for i in range(self.num_layers)
#             ]

#         outputs = []
#         for t in range(seq_len):
#             x = inputs[:, t, :]
#             new_hidden = []
#             for i, cell in enumerate(self.cells):
#                 x, new_state = cell(x, hidden_state[i])
#                 new_hidden.append(new_state)
#             hidden_state = new_hidden
#             outputs.append(x.unsqueeze(1))  # keep time dimension

#         outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, num_nodes * output_dim)
#         outputs = self.fc_out(outputs)
#         return outputs, hidden_state



# 注意力机制的DCGRU
class DCGRUCell(nn.Module):
    """Graph Convolution Gated Recurrent Unit cell - PyTorch implementation."""
    
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, input_dim, num_proj=None,
                 activation=torch.tanh, filter_type="laplacian", use_gc_for_ru=True, use_attention=False):
        super(DCGRUCell, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        self._use_attention = use_attention

        # Build support matrices
        self._supports = []
        supports = []

        if filter_type == "laplacian":
            supports.append(self._calculate_scaled_laplacian(adj_mx))
        elif filter_type == "random_walk":
            supports.append(self._calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(self._calculate_random_walk_matrix(adj_mx).T)
            supports.append(self._calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(self._calculate_scaled_laplacian(adj_mx))

        # Convert to sparse tensors
        for support in supports:
            self._supports.append(self._build_sparse_tensor(support))

        # Calculate dimensions for weight matrices
        num_matrices = len(self._supports) * self._max_diffusion_step + 1

        # Initialize weight matrices
        if self._use_gc_for_ru:
            # Graph convolution weights for reset and update gates
            self.gate_weights = nn.Parameter(
                torch.empty(num_matrices * (input_dim + num_units), 2 * num_units)
            )
            nn.init.xavier_uniform_(self.gate_weights)
            self.gate_biases = nn.Parameter(torch.ones(2 * num_units))  # Bias of 1.0 for GRU
        else:
            # Fully connected weights for reset and update gates
            self.fc_gate_weights = nn.Parameter(
                torch.empty(2 * num_units, 2 * num_units)
            )
            nn.init.xavier_uniform_(self.fc_gate_weights)
            self.fc_gate_biases = nn.Parameter(torch.ones(2 * num_units))

        # Graph convolution weights for candidate state
        self.candidate_weights = nn.Parameter(
            torch.empty(num_matrices * (input_dim + num_units), num_units)
        )
        nn.init.xavier_uniform_(self.candidate_weights)
        self.candidate_biases = nn.Parameter(torch.zeros(num_units))

        # Attention mechanism
        if self._use_attention:
            self.attention_weights = nn.Parameter(torch.empty(num_nodes, num_nodes))
            nn.init.xavier_uniform_(self.attention_weights)

        # Projection weights (optional)
        if self._num_proj is not None:
            self.proj_weights = nn.Parameter(torch.empty(num_units, num_proj))
            nn.init.xavier_uniform_(self.proj_weights)
    
    @staticmethod
    def _calculate_scaled_laplacian(adj_mx, lambda_max=2.0):
        """Calculate scaled Laplacian matrix."""
        # adj_mx = sp.coo_matrix(adj_mx)
        
        adj_tensor = adj_mx[0] if isinstance(adj_mx, list) else adj_mx
        adj_mx = sp.coo_matrix(adj_tensor.cpu().numpy())


        d = np.array(adj_mx.sum(1)).flatten()
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj_mx.shape[0]) - d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt)
        
        if lambda_max is None:
            lambda_max = sp.linalg.eigsh(normalized_laplacian, 1, which='LA', return_eigenvectors=False)[0]
        
        return (2 / lambda_max * normalized_laplacian - sp.eye(normalized_laplacian.shape[0])).astype(np.float32)
    
    @staticmethod
    def _calculate_random_walk_matrix(adj_mx):
        """Calculate random walk matrix."""
        adj_mx = sp.coo_matrix(adj_mx)
        d = np.array(adj_mx.sum(1)).flatten()
        d_inv = np.power(d, -1)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
        return random_walk_mx.astype(np.float32)
    
    def _build_sparse_tensor(self, sparse_mx):
        """Convert scipy sparse matrix to PyTorch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)
    
    @property
    def state_size(self):
        return self._num_nodes * self._num_units
    
    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size
    
    def _gconv(self, inputs, state, output_size, bias_start=0.0, weights=None, biases=None):
        """Graph convolution between input and the graph matrix."""
        batch_size = inputs.size(0)

        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        inputs = inputs.view(batch_size, self._num_nodes, -1)
        state = state.view(batch_size, self._num_nodes, -1)
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        # Apply attention mechanism if enabled
        if self._use_attention:
            attention = torch.softmax(self.attention_weights, dim=-1)
            inputs_and_state = torch.matmul(inputs_and_state, attention)

        # Transpose for graph convolution: (num_nodes, input_size, batch_size)
        x = inputs_and_state.permute(1, 2, 0)
        x0 = x.contiguous().view(self._num_nodes, input_size * batch_size)
        x_list = [x0]

        # Apply diffusion steps
        if self._max_diffusion_step > 0:
            for support in self._supports:
                support = support.to(x0.device)
                x1 = torch.sparse.mm(support, x0)
                x_list.append(x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x_list.append(x2)
                    x1, x0 = x2, x1

        # Stack all diffusion results
        num_matrices = len(x_list)
        x = torch.stack(x_list, dim=0)  # (num_matrices, num_nodes, input_size * batch_size)
        x = x.view(num_matrices, self._num_nodes, input_size, batch_size)
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, num_matrices)
        x = x.contiguous().view(batch_size * self._num_nodes, input_size * num_matrices)

        # Apply linear transformation
        x = torch.matmul(x, weights) + biases

        # Reshape back to (batch_size, num_nodes * output_size)
        return x.view(batch_size, self._num_nodes * output_size)
    

    
    def _fc(self, inputs, state, output_size):
        """Fully connected layer for gates."""
        batch_size = inputs.size(0)
        inputs = inputs.view(batch_size * self._num_nodes, -1)
        state = state.view(batch_size * self._num_nodes, -1)
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        
        output = torch.matmul(inputs_and_state, self.fc_gate_weights) + self.fc_gate_biases
        return output
    
    def forward(self, inputs, state):
        """
        Forward pass of DCGRU cell.
        
        Args:
            inputs: Input tensor of shape (batch_size, num_nodes * input_dim)
            state: Hidden state tensor of shape (batch_size, num_nodes * num_units)
            
        Returns:
            output: Output tensor
            new_state: New hidden state
        """
        batch_size = inputs.size(0)
        
        # Calculate gates (reset and update)
        if self._use_gc_for_ru:
            gate_inputs = self._gconv(inputs, state, 2 * self._num_units, 
                                    bias_start=1.0, weights=self.gate_weights, biases=self.gate_biases)
        else:
            gate_inputs = self._fc(inputs, state, 2 * self._num_units)
        
        gate_inputs = torch.sigmoid(gate_inputs)
        gate_inputs = gate_inputs.view(batch_size, self._num_nodes, 2 * self._num_units)
        r, u = torch.split(gate_inputs, self._num_units, dim=-1)
        r = r.contiguous().view(batch_size, self._num_nodes * self._num_units)
        u = u.contiguous().view(batch_size, self._num_nodes * self._num_units)
        
        # Calculate candidate state
        candidate = self._gconv(inputs, r * state, self._num_units, 
                              weights=self.candidate_weights, biases=self.candidate_biases)
        if self._activation is not None:
            candidate = self._activation(candidate)
        
        # Calculate new state
        new_state = u * state + (1 - u) * candidate
        output = new_state
        
        # Apply projection if specified
        if self._num_proj is not None:
            output = output.view(batch_size * self._num_nodes, self._num_units)
            output = torch.matmul(output, self.proj_weights)
            output = output.view(batch_size, self.output_size)
        
        return output, new_state


# 增加残差
class DCGRU(nn.Module):
    """Multi-layer DCGRU model."""
    
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes,
                 input_dim, output_dim, num_layers=1, num_proj=None,
                 activation=torch.tanh, filter_type="laplacian", use_gc_for_ru=True):
        super(DCGRU, self).__init__()
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim

        cells = []
        for i in range(num_layers):
            cell_input_dim = input_dim if i == 0 else num_units
            cell = DCGRUCell(
                num_units=num_units,
                adj_mx=adj_mx,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                input_dim=cell_input_dim,
                num_proj=num_proj if i == num_layers - 1 else None,
                activation=activation,
                filter_type=filter_type,
                use_gc_for_ru=use_gc_for_ru
            )
            cells.append(cell)
        self.cells = nn.ModuleList(cells)

        # Optional final fully connected layer to map to output_dim
        if num_proj is not None:
            self.fc_out = nn.Linear(num_proj * num_nodes, output_dim * num_nodes)
        else:
            self.fc_out = nn.Linear(num_units * num_nodes, output_dim * num_nodes)

    def forward(self, inputs, hidden_state=None):
        """
        Args:
            inputs: Input tensor (batch_size, seq_len, num_nodes * input_dim)
            hidden_state: Optional list of hidden states per layer

        Returns:
            outputs: Tensor of shape (batch_size, seq_len, num_nodes * output_dim)
            final_state: Final hidden states
        """
        batch_size, seq_len, _ = inputs.shape
        if hidden_state is None:
            hidden_state = [
                torch.zeros(batch_size, self.cells[i].state_size, device=inputs.device)
                for i in range(self.num_layers)
            ]
        inputs_reshaped = inputs 

        outputs = []
        for t in range(seq_len):
            # x = inputs[:, t, :]
            x = inputs_reshaped[:, t, :]
            new_hidden = []
            for i, cell in enumerate(self.cells):
                x, new_state = cell(x, hidden_state[i])
                new_hidden.append(new_state)
            hidden_state = new_hidden
            outputs.append(x.unsqueeze(1))  # keep time dimension

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, num_nodes * output_dim)
        outputs = self.fc_out(outputs)

        # print("outputs shape:", outputs.shape)  # 打印 outputs 的形状
        # print("inputs shape:", inputs.shape)   # 打印 inputs 的形状
            
        # Ensure output dimension matches input dimension for residual connection
        if outputs.shape[-1] != inputs_reshaped.shape[-1]:
            # Add a linear projection if dimensions don't match
            adjust_proj = nn.Linear(outputs.shape[-1], inputs_reshaped.shape[-1]).to(outputs.device)
            outputs = adjust_proj(outputs)
        # if outputs.shape[-1] != inputs.shape[-1]:
        #     raise ValueError(f"Output shape {outputs.shape} does not match input shape {inputs.shape}")        
        
        # return outputs + inputs, hidden_state  # 添加残差连接
        return outputs + inputs_reshaped, hidden_state  # 添加残差连接
