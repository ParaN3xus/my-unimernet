from transformers.models.mbart.modeling_mbart import MBartConfig, MBartDecoderLayer
from .attn import MBART_ATTENTION_CLASSES


class UniMERNetDecoderLayer(MBartDecoderLayer):
    def __init__(self, config: MBartConfig):
        super().__init__(config)

        self.self_attn = MBART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
