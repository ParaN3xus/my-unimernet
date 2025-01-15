from transformers import SwinModel, DonutSwinConfig
from transformers.models.swin.modeling_swin import SwinEncoder, SwinPatchMerging
from torch import nn
from .embeddings.embeddings import UniMERNetEmbeddings
from .layers.stage import UniMERNetEncoderStage
import torch

# should remove layernorm


class UniMERNetEncoderModel(SwinModel):
    config_class = DonutSwinConfig

    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        super().__init__(config, add_pooling_layer, use_mask_token)

        self.embeddings = UniMERNetEmbeddings(
            config, use_mask_token=use_mask_token)
        self.encoder = UniMERNetEncoder(config, self.embeddings.patch_grid)

        # Initialize weights and apply final processing
        self.post_init()


class UniMERNetEncoder(SwinEncoder):
    def __init__(self, config, grid_size):
        super().__init__(config, grid_size)

        dpr = [x.item() for x in torch.linspace(
            0, config.drop_path_rate, sum(config.depths))]
        self.layers = nn.ModuleList(
            [
                UniMERNetEncoderStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(
                        grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[sum(config.depths[:i_layer]): sum(
                        config.depths[: i_layer + 1])],
                    downsample=SwinPatchMerging if (
                        i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False
