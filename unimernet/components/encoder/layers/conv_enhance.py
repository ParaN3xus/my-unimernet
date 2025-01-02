from torch import nn
from typing import Tuple
from transformers.activations import ACT2FN


class ConvEnhance(nn.Module):
    """Depth-wise convolution to get the positional information.
    """

    def __init__(self, config, dim, k=3):
        super(ConvEnhance, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              (k, k),
                              (1, 1),
                              (k // 2, k // 2),
                              groups=dim)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        feat = self.act_fn(feat)
        feat = feat.flatten(2).transpose(1, 2)

        x = x + feat
        return x
