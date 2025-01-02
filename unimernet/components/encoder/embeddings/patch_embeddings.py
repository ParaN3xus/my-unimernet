from transformers.models.swin.modeling_swin import SwinPatchEmbeddings
from fge import StemLayer


class UnimerNetPatchEmbeddings(SwinPatchEmbeddings):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        print("VariableUnimerNetPatchEmbeddings init")
        super().__init__(config)
        num_channels, hidden_size = config.num_channels, config.embed_dim
        self.projection = StemLayer(
            in_chans=num_channels, out_chans=hidden_size)
