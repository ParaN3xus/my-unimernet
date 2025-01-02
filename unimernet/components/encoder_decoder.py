import torch
import torch.nn as nn
from transformers import AutoModel, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from configuration import UnimerNetConfig
from encoder.encoder import UnimerNetEncoder
from decoder.decoder import UnimerNetDecoder


class UnimerNetEncoderDecoder(nn.Module):

    def __init__(self, model_name, num_tokens, pad_token_id, bos_token_id, eos_token_id):
        super().__init__()
        config = VisionEncoderDecoderConfig.from_pretrained(model_name)
        encoder_config = vars(config.encoder)
        encoder = UnimerNetConfig(**encoder_config)
        config.encoder = encoder
        self.config = config

        AutoModel.register(UnimerNetConfig, UnimerNetEncoder)

        self.model = CustomVisionEncoderDecoderModel(config=self.config)

        self.model.config.decoder_start_token_id = bos_token_id
        self.model.config.pad_token_id = pad_token_id
        self.model.config.eos_token_id = eos_token_id
        self.model.decoder.resize_token_embeddings(num_tokens)
        self.pad_token_id = pad_token_id

    def forward(self, pixel_values, decoder_input_ids, decoder_attention_mask, **kwargs):
        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)

        labels = decoder_input_ids * 1
        labels = labels.masked_fill(labels == self.pad_token_id, -100)

        loss = self.model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids[:, :-1],
            decoder_attention_mask=decoder_attention_mask[:, :-1],
            labels=labels[:, 1:],
            **kwargs
        ).loss
        return loss

    @torch.no_grad()
    def generate(self, pixel_values, temperature, max_new_tokens, decoder_start_token_id, do_sample, top_p,
                 **kwargs):

        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)
        outputs = self.model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            decoder_start_token_id=decoder_start_token_id,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
        )
        return outputs[:, 1:]


class CustomVisionEncoderDecoderModel(VisionEncoderDecoderModel):
    def __init__(self, config):
        print("CustomVisionEncoderDecoderModel init")
        super().__init__(config)
        # Replace the MBartForCausalLM with your CustomMBartForCausalLM
        self.encoder = UnimerNetEncoder(config.encoder)
        self.decoder = UnimerNetDecoder(self.config.decoder)
