from typing import Optional, Tuple
from transformers.models.swin.modeling_swin import SwinEmbeddings
from torch import nn
import torch
from .patch_embeddings import UniMERNetPatchEmbeddings


class UniMERNetEmbeddings(SwinEmbeddings):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__(config, use_mask_token)

        self.patch_embeddings = UniMERNetPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.mask_token = nn.Parameter(torch.zeros(
            1, 1, config.embed_dim)) if use_mask_token else None
        self.position_embeddings = None

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, num_patches + 1, config.embed_dim))

        self.row_embeddings = None
        self.column_embeddings = None
        if config.use_2d_embeddings:
            self.row_embeddings = nn.Parameter(torch.zeros(
                1, self.patch_grid[0] + 1, config.embed_dim))
            self.column_embeddings = nn.Parameter(torch.zeros(
                1, self.patch_grid[1] + 1, config.embed_dim))

        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor] = None, interpolate_pos_encoding: bool = False,
    ) -> Tuple[torch.Tensor]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)

        # Layernorm across the last dimension (each patch is a single row)
        embeddings = self.norm(embeddings)
        batch_size, seq_len, embed_dim = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings[:, :seq_len, :]

        if self.row_embeddings is not None and self.column_embeddings is not None:
            # Repeat the x position embeddings across the y axis like 0, 1, 2, 3, 0, 1, 2, 3, ...
            row_embeddings = self.row_embeddings[:, :output_dimensions[0], :].repeat_interleave(output_dimensions[1],
                                                                                                dim=1)
            column_embeddings = self.column_embeddings[:, :output_dimensions[1], :].repeat(
                1, output_dimensions[0], 1)

            embeddings = embeddings + row_embeddings + column_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions
