import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

class SpatialAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.heads = heads
        self.d_head = channels // heads
        # one 1×1 conv to produce Q||K||V
        self.qkv_conv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        # project to Q,K,V
        qkv = self.qkv_conv(x)                     # (N, 3C, T, V)
        Q, K, Vv = torch.split(qkv, C, dim=1)       # each (N, C, T, V)

        # reshape & permute to (N, heads, T, V, d_head)
        def to_heads(y):
            y = y.view(N, self.heads, self.d_head, T, V)
            return y.permute(0, 1, 3, 4, 2)

        Qh = to_heads(Q)
        Kh = to_heads(K)
        Vh = to_heads(Vv)

        # mask out non-edges
        adj = A.any(dim=0)  # → shape (V, V)
        mask = (~adj).view(1, 1, 1, V, V)

        # fused FlashAttention call
        out = F.scaled_dot_product_attention(
            Qh, Kh, Vh,
            attn_mask=mask,
            dropout_p=self.dropout.p,
            is_causal=False
        )  # → (N, heads, T, V, d_head)

        # back to (N, C, T, V)
        out = out.permute(0,1,4,2,3).contiguous()   # (N, heads, d_head, T, V)
        return out.view(N, C, T, V)


class TemporalAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert channels % heads == 0
        self.heads  = heads
        self.d_head = channels // heads
        self.qkv_conv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        qkv = self.qkv_conv(x)                      # (N, 3C, T, V)
        Q, K, Vv = torch.split(qkv, C, dim=1)        # each (N, C, T, V)

        # reshape so time is attention axis: (N, heads, V, T, d_head)
        def to_heads_time(y):
            y = y.view(N, self.heads, self.d_head, T, V)
            return y.permute(0,1,4,3,2)

        Qh = to_heads_time(Q)
        Kh = to_heads_time(K)
        Vh = to_heads_time(Vv)

        # fused FlashAttention over frames
        out = F.scaled_dot_product_attention(
            Qh, Kh, Vh,
            attn_mask=None,
            dropout_p=self.dropout.p,
            is_causal=False
        )  # → (N, heads, V, T, d_head)

        # back to (N, C, T, V)
        out = out.permute(0,1,4,3,2).contiguous()   # (N, heads, d_head, T, V)
        return out.view(N, C, T, V)

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

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)

        # prediction
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        return x


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
        #x = self.spatial_attn(x, A)
        #x = self.temporal_attn(x)
        x = x + self.spatial_attn(x, A)
        x = x + self.temporal_attn(x)
        x = self.tcn(x) + res

        return self.relu(x), A