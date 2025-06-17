import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

class SpatialAttention(nn.Module):
    """
    Spatial multi‐head attention over V joints at each of T frames,
    using three separate 1×1 convs for Q, K, and V. Returns (out, attn_weights).

    Input:
      x: (N, C, T, V)
      A: (K, V, V) or (V, V) adjacency
    Output:
      out:  (N, C, T, V)
      attn: (N, heads, T, V, V)
    """
    def __init__(self, channels: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.heads = heads
        self.d_head = channels // heads

        # three separate 1×1 convs for Q, K, V
        self.q_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        N, C, T, V = x.shape

        # 1) project to Q, K, V separately
        Q = self.q_conv(x)    # (N, C, T, V)
        K = self.k_conv(x)    # (N, C, T, V)
        Vv= self.v_conv(x)    # (N, C, T, V)

        # 2) reshape & permute into heads → (N, heads, T, V, d_head)
        def to_heads(y):
            y = y.view(N, self.heads, self.d_head, T, V)
            return y.permute(0, 1, 3, 4, 2)

        Qh = to_heads(Q)
        Kh = to_heads(K)
        Vh = to_heads(Vv)

        # 3) build adjacency mask: collapse (K, V, V)→(V, V) if needed
        adj = A.any(dim=0)    # (V, V) boolean

        mask = (~adj).view(1, 1, 1, V, V)  # (1,1,1,V,V) broadcastable to (N,heads,T,V,V)

        # 4) compute raw attention scores → (N, heads, T, V, V)
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(mask, float("-inf"))

        # 5) softmax + dropout → attn weights
        attn = F.softmax(scores, dim=-1)            # (N, heads, T, V, V)
        attn = self.dropout(attn)

        # 6) aggregate values → (N, heads, T, V, d_head)
        out_h = torch.matmul(attn, Vh)

        # 7) recombine heads → (N, C, T, V)
        out = out_h.permute(0, 1, 4, 2, 3).contiguous()  # (N, heads, d_head, T, V)
        out = out.view(N, C, T, V)

        return out, attn

class TemporalAttention(nn.Module):
    """
    Non‐causal temporal multi‐head attention over T frames for each of V joints,
    using three separate 1×1 convs for Q, K, and V. Returns (out, attn_weights).

    Input:
      x: (N, C, T, V)
    Output:
      out:  (N, C, T, V)
      attn: (N, heads, V, T, T)
    """
    def __init__(self, channels: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.heads  = heads
        self.d_head = channels // heads

        # three separate 1×1 convs for Q, K, V
        self.q_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        N, C, T, V = x.shape

        # 1) project to Q, K, V separately
        Q = self.q_conv(x)    # (N, C, T, V)
        K = self.k_conv(x)    # (N, C, T, V)
        Vv= self.v_conv(x)    # (N, C, T, V)

        # 2) reshape & permute so time is the attention axis → (N, heads, V, T, d_head)
        def to_heads_time(y):
            y = y.view(N, self.heads, self.d_head, T, V)
            return y.permute(0, 1, 4, 3, 2)  # (N, heads, V, T, d_head)

        Qh = to_heads_time(Q)
        Kh = to_heads_time(K)
        Vh = to_heads_time(Vv)

        # 3) compute raw attention scores → (N, heads, V, T, T)
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 4) softmax + dropout → attn weights
        attn = F.softmax(scores, dim=-1)             # (N, heads, V, T, T)
        attn = self.dropout(attn)

        # 5) aggregate values → (N, heads, V, T, d_head)
        out_h = torch.matmul(attn, Vh)

        # 6) recombine heads → (N, C, T, V)
        out = out_h.permute(0, 1, 4, 3, 2).contiguous()  # (N, heads, d_head, T, V)
        out = out.view(N, C, T, V)

        return out, attn

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))
        self.fc = nn.Linear(hidden_dim, num_class)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        all_sp_attns = []
        all_tp_attns = []

        # 2) ST‐GCN + collect attention weights
        for gcn, imp in zip(self.st_gcn_networks, self.edge_importance):
            x, _, sp_attn, tp_attn = gcn(x, self.A * imp)
            all_sp_attns.append(sp_attn.detach())  # (N*M, heads, T, V, V)
            all_tp_attns.append(tp_attn.detach())  # (N*M, heads, V, T, T)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)

        # prediction
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        return x, all_sp_attns, all_tp_attns


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 attn_heads: int = 8,
                 attn_dropout: float = 0.1):

        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        # FlashAttention modules
        self.spatial_attn = SpatialAttention(out_channels,
                                                  heads=attn_heads,
                                                  dropout=attn_dropout)
        self.temporal_attn = TemporalAttention(out_channels,
                                                    heads=attn_heads,
                                                    dropout=attn_dropout)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        out_sp, att_sp = self.spatial_attn(x, A)
        x = x + out_sp

        out_tp, att_tp = self.temporal_attn(x)
        x = x + out_tp

        x = self.tcn(x) + res
        return self.relu(x), A, att_sp, att_tp
