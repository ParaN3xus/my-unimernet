from transformers import DonutSwinConfig, MBartConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class UnimerNetConfig(DonutSwinConfig):
    """
    UniMerNet Encoder Config
    """
    model_type = "donut-swin"
