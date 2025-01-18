from transformers import SwinModel, DonutSwinConfig


class UniMERNetEncoderConfig(DonutSwinConfig):
    model_type = "unimernet_encoder"
