import logging
from omegaconf import OmegaConf
import cv2
from PIL import Image, ImageOps
from torchvision.transforms.functional import resize
import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger()

IMAGE_STD = [0.229, 0.224, 0.225]
IMAGE_MEAN = [0.485, 0.456, 0.406]


class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    def build(self, **kwargs):
        cfg = OmegaConf.create(kwargs)

        return self.from_config(cfg)


class FormulaImageBaseProcessor(BaseProcessor):

    def __init__(self, image_size):
        super(FormulaImageBaseProcessor, self).__init__()
        self.input_size = [int(_) for _ in image_size]
        assert len(self.input_size) == 2

    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        # Find minimum spanning bounding box
        a, b, w, h = cv2.boundingRect(coords)
        return img.crop((a, b, w + a, h + b))

    def prepare_input(self, img: Image.Image, random_padding: bool = False):
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        if img is None:
            return
        # crop margins
        try:
            img = self.crop_margin(img.convert("RGB"))
        except OSError:
            # might throw an error for broken files
            return

        if img.height == 0 or img.width == 0:
            return

        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return ImageOps.expand(img, padding)


class FormulaImageEvalProcessor(FormulaImageBaseProcessor):
    def __init__(self, image_size):
        super().__init__(image_size)

        self.transform = alb.Compose(
            [
                alb.ToGray(always_apply=True),
                alb.Normalize((0.7931, 0.7931, 0.7931),
                              (0.1738, 0.1738, 0.1738)),
                # alb.Sharpen()
                ToTensorV2(),
            ]
        )

    def __call__(self, item):
        image = self.prepare_input(item)
        return self.transform(image=np.array(image))['image'][:1]

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", [384, 384])

        return cls(image_size=image_size)
