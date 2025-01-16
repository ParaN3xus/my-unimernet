# My UniMERNet
A cleaner and more organized version of [UniMERNet](https://github.com/opendatalab/UniMERNet).

## Features
- Maximizes code reuse through inheritance from `Swin` and `MBart` in [transformers](https://github.com/huggingface/transformers) library, avoiding extensive code duplication
- Use transformers-compatible interfaces (`VisionEncoderDecoder`, `TrOCRProcessor`, etc.)
- Removes non-UniMERNET components

## Usage
Convert a official model with [convert.ipynb](https://github.com/ParaN3xus/my-unimernet/blob/main/utils/convert.ipynb) and then use it like `transformers.VisionEncoderDecoderModel`, just like [TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr#inference).

## Credits
Besides direct Python imports, this project makes use of the following open-source projects:

- [tramsformers](https://github.com/huggingface/transformers): State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.
- [UniMERNet](https://github.com/opendatalab/UniMERNet): A Universal Network for Real-World Mathematical Expression Recognition.

## LICENSE
This repository is published under an MIT License. See [LICENSE](https://github.com/ParaN3xus/my-unimernet/blob/main/LICENSE) file.