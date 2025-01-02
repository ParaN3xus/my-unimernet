from transformers import SwinModel
from transformers.models.swin.modeling_swin import SwinEncoder, SwinPatchMerging
from configuration import UnimerNetConfig
from torch import nn
from embeddings.embeddings import UnimerNetEmbeddings
from layers.stage import UnimerNetEncoderStage


class UnimerNetEncoder(SwinModel):
    config_class = UnimerNetConfig

    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        print("VariableUnimerNetModel init")
        super().__init__(config)

        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        self.embeddings = UnimerNetEmbeddings(
            config, use_mask_token=use_mask_token)
        self.encoder = UnimerNetEncoder(config, self.embeddings.patch_grid)

        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


class UnimerNetEncoder(SwinEncoder):
    def __init__(self, config, grid_size):
        super().__init__(config, grid_size)
        self.layers = nn.ModuleList(
            [
                UnimerNetEncoderStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(
                        grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=self.dpr[sum(config.depths[:i_layer]): sum(
                        config.depths[: i_layer + 1])],
                    downsample=SwinPatchMerging if (
                        i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False
