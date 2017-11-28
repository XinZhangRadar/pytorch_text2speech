##
PyTorch implementation of convolutional networks-based text-to-speech synthesis models:

1. [arXiv:1710.07654](https://arxiv.org/abs/1710.07654): Deep Voice 3: 2000-Speaker Neural Text-to-Speech.
2. [arXiv:1710.08969](https://arxiv.org/abs/1710.08969): Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention.

## Requirements

- Python 3
- PyTorch >= v0.2 (Note: I'm using v0.3.0 branch)
- TensorFlow >= v1.3
- [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) (master)
- [fairseq](https://github.com/facebookresearch/fairseq-py) (master)
- [nnmnkwii](https://github.com/r9y9/nnmnkwii) >= v0.0.9
- [MeCab](http://taku910.github.io/mecab/) (Japanese only)


## Getting started

### 0. Download dataset

- LJSpeech (en): https://keithito.com/LJ-Speech-Dataset/
- JSUT (jp): https://sites.google.com/site/shinnosuketakamichi/publication/jsut

### 1. Preprocessing

Preprocessing can be done by `preprocess.py`. Usage is:

```
python preprocess.py ${dataset_name} ${dataset_path} ${out_dir}
```

Supported `${dataset_name}`s for now are `ljspeech` and `jsut`. Suppose you will want to preprocess LJSpeech dataset and have it in `~/data/LJSpeech-1.0`, then you can preprocess data by:

```
python preprocess.py ljspeech ~/data/LJSpeech-1.0/ ./data/ljspeech
```

When this is done, you will see extracted features (mel-spectrograms and linear spectrograms) in `./data/ljspeech`.

### 2. Training

Basic usage of `train.py` is:

```
python train.py --data-root=${data-root} --hparams="parameters you want to override"
```

Suppose you will want to build a Deep Voice 3-style model using LJSpeech dataset with default hyper parameters, then you can train your model by:

```
python train.py --data-root=./data/ljspeech/ --hparams="use_preset=True,builder=text2speech"
```

Model checkpoints (.pth) and alignments (.png) are saved in `./checkpoints` directory per 5000 steps by default.

If you are building a Japaneses TTS model, then for example,

```
python train.py --data-root=./data/jsut --hparams="frontend=jp" --hparams="use_preset=True,builder=text2speech"
```

`frontend=jp` tell the training script to use Japanese text processing frontend. Default is `en` and uses English text processing frontend.

Note that there are many hyper parameters and design choices. Some are configurable by `hparams.py` and some are hardcoded in the source (e.g., dilation factor for each convolution layer). If you find better hyper parameters, please let me know!


### 4. Moniter with Tensorboard

Logs are dumped in `./log` directory by default. You can monitor logs by tensorboard:

```
tensorboard --logdir=log
```

### 5. Synthesize from a checkpoint

Given a list of text, `synthesis.py` synthesize audio signals from trained model. Usage is:

```
python synthesis.py ${checkpoint_path} ${text_list.txt} ${output_dir}
```

Example test_list.txt:

```
Generative adversarial network or variational auto-encoder.
Once upon a time there was a dear little girl who was loved by every one who looked at her, but most of all by her grandmother, and there was nothing that she would not have given to the child.
A text-to-speech synthesis system typically consists of multiple stages, such as a text analysis frontend, an acoustic model and an audio synthesis module.
```

## Acknowledgements

Part of code was adapted from the following projects:

- https://github.com/keithito/tacotron
- https://github.com/facebookresearch/fairseq-py
