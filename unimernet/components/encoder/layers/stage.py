from torch import nn
from transformers.models.swin.modeling_swin import SwinStage
from layer import UnimerNetEncoderLayer

# Copied from transformers.models.swin.modeling_swin.SwinStage with Swin->UnimerNet


class UnimerNetEncoderStage(SwinStage):
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        super().__init__(config, dim, input_resolution,
                         depth, num_heads, drop_path, downsample)
        self.blocks = nn.ModuleList(
            [
                UnimerNetEncoderLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads
                )
                for i in range(depth)
            ]
        )
