from .components.decoder.decoder import UniMERNetDecoderModel, UniMERNetDecoderConfig
from .components.encoder.encoder import UniMERNetEncoderModel, UniMERNetEncoderConfig
from .components.processor.image_processor import UniMERNetEvalImageProcessor, UniMERNetTrainImageProcessor
from .components.processor.config import UniMERNetEvalImageProcessorConfig, UniMERNetTrainImageProcessorConfig

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoImageProcessor

AutoConfig.register("unimernet_encoder", UniMERNetEncoderConfig)
AutoModel.register(UniMERNetEncoderConfig, UniMERNetEncoderModel)

AutoConfig.register("unimernet_decoder", UniMERNetDecoderConfig)
AutoModelForCausalLM.register(UniMERNetDecoderConfig, UniMERNetDecoderModel)

AutoConfig.register("unimernet_evalprocessor",
                    UniMERNetEvalImageProcessorConfig)
AutoImageProcessor.register(
    UniMERNetEvalImageProcessorConfig,
    slow_image_processor_class=UniMERNetEvalImageProcessor,
    fast_image_processor_class=None)

AutoConfig.register("unimernet_trainprocessor",
                    UniMERNetTrainImageProcessorConfig)
AutoImageProcessor.register(
    UniMERNetTrainImageProcessorConfig,
    slow_image_processor_class=UniMERNetTrainImageProcessor,
    fast_image_processor_class=None)


__all__ = ["UniMERNetDecoderModel",
           "UniMERNetDecoderConfig",
           "UniMERNetEncoderModel",
           "UniMERNetEncoderConfig",
           "UniMERNetEvalImageProcessor",
           "UniMERNetTrainImageProcessor",
           "UniMERNetEvalImageProcessorConfig",
           "UniMERNetTrainImageProcessorConfig"]
