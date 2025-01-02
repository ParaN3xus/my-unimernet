from typing import Tuple, Optional
import torch
from torch import nn
from transformers.models.swin.modeling_swin import SwinLayer, window_partition, window_reverse
from transformers.utils import torch_int
from conv_enhance import ConvEnhance

# Copied from transformers.models.swin.modeling_swin.SwinLayer with Swin->UnimerNet


class UnimerNetEncoderLayer(SwinLayer):
    def __init__(self, config, dim, input_resolution, num_heads):
        super().__init__(config, dim, input_resolution, num_heads, shift_size=0)

        self.ce = nn.ModuleList([ConvEnhance(config, dim=dim, k=3),
                                 ConvEnhance(config, dim=dim, k=3)])

    def set_shift_and_window_size(self, input_resolution):
        if min(input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = torch_int(0)
            self.window_size = (
                torch.min(torch.tensor(input_resolution)
                          ) if torch.jit.is_tracing() else min(input_resolution)
            )

    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width %
                     self.window_size) % self.window_size
        pad_bottom = (self.window_size - height %
                      self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        else:
            pass
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.size()

        hidden_states = self.ce[0](
            hidden_states, input_dimensions)
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(
            hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape

        # no more cyclic shift
        shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(
            shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(
            -1, self.window_size * self.window_size, channels)

        attention_outputs = self.attention(
            hidden_states_windows, None, head_mask, output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(
            -1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(
            attention_windows, self.window_size, height_pad, width_pad)

        # no more reverse cyclic shift
        attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:,
                                                  :height, :width, :].contiguous()

        attention_windows = attention_windows.view(
            batch_size, height * width, channels)

        hidden_states = shortcut + self.drop_path(attention_windows)

        hidden_states = self.ce[1](hidden_states, input_dimensions)
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output)

        layer_outputs = (
            layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs
