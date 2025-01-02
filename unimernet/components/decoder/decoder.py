from transformers import MBartForCausalLM
from transformers.models.mbart.modeling_mbart import MBartDecoder
from torch import nn
from layers.layer import UnimerNetDecoderLayer


class UnimerNetDecoder(MBartForCausalLM):
    def __init__(self, config):
        print("CustomMBartForCausalLM init")
        super().__init__(config)
        # Modify the decoder within MBartDecoderWrapper
        self.model.decoder = UnimerNetMBartDecoder(config)


class UnimerNetMBartDecoder(MBartDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [UnimerNetDecoderLayer(config) for _ in range(config.decoder_layers)])
