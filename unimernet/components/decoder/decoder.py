from transformers import MBartForCausalLM
from transformers.models.mbart.modeling_mbart import MBartDecoder
from torch import nn
from .layers.layer import UniMERNetDecoderLayer


class UniMERNetDecoderModel(MBartForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # Modify the decoder within MBartDecoderWrapper
        self.model.decoder = UniMERNetDecoder(config)


class UniMERNetDecoder(MBartDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [UniMERNetDecoderLayer(config) for _ in range(config.decoder_layers)])
