<h2 align="center">
<p> :yum: TensorFlowTTS
<p align="center">
    <a href="https://github.com/tensorspeech/TensorFlowTTS/actions">
        <img alt="Build" src="https://github.com/tensorspeech/TensorFlowTTS/workflows/CI/badge.svg?branch=master">
    </a>
    <a href="https://github.com/tensorspeech/TensorFlowTTS/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/tensorspeech/TensorflowTTS?color=red">
    </a>
    <a href="https://colab.research.google.com/drive/1akxtrLZHKuMiQup00tzO2olCaN-y3KiD?usp=sharing">
        <img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
</p>
</h2>
<h2 align="center">
<p>Real-Time State-of-the-art Speech Synthesis for Tensorflow 2
</h2>

:zany_face: TensorFlowTTS provides real-time state-of-the-art speech synthesis architectures such as Tacotron-2, Melgan, Multiband-Melgan, FastSpeech, FastSpeech2 based-on TensorFlow 2. With Tensorflow 2, we can speed-up training/inference progress, optimizer further by using [fake-quantize aware](https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide) and [pruning](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras), make TTS models can be run faster than real-time and be able to deploy on mobile devices or embedded systems.

## What's new
- 2021/08/18 (**NEW!**) Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/TensorFlowTTS).
- 2021/08/12 (**NEW!**) Support French TTS (Tacotron2, Multiband MelGAN). Pls see the [colab](https://colab.research.google.com/drive/1jd3u46g-fGQw0rre8fIwWM9heJvrV1c0?usp=sharing). Many Thanks [Samuel Delalez](https://github.com/samuel-lunii)
- 2021/06/01 Integrated with [Huggingface Hub](https://huggingface.co/tensorspeech). See the [PR](https://github.com/TensorSpeech/TensorFlowTTS/pull/555). Thanks [patrickvonplaten](https://github.com/patrickvonplaten) and [osanseviero](https://github.com/osanseviero)
- 2021/03/18  Support IOS for FastSpeech2 and MB MelGAN. Thanks [kewlbear](https://github.com/kewlbear). See [here](https://github.com/TensorSpeech/TensorFlowTTS/tree/master/examples/ios)
- 2021/01/18 Support TFLite C++ inference. Thanks [luan78zaoha](https://github.com/luan78zaoha). See [here](https://github.com/TensorSpeech/TensorFlowTTS/tree/master/examples/cpptflite)
- 2020/12/02 Support German TTS with [Thorsten dataset](https://github.com/thorstenMueller/deep-learning-german-tts). See the [Colab](https://colab.research.google.com/drive/1W0nSFpsz32M0OcIkY9uMOiGrLTPKVhTy?usp=sharing). Thanks [thorstenMueller](https://github.com/thorstenMueller) and [monatis](https://github.com/monatis)
- 2020/11/24 Add HiFi-GAN vocoder. See [here](https://github.com/TensorSpeech/TensorFlowTTS/tree/master/examples/hifigan)
- 2020/11/19 Add Multi-GPU gradient accumulator. See [here](https://github.com/TensorSpeech/TensorFlowTTS/pull/377)
- 2020/08/23 Add Parallel WaveGAN tensorflow implementation. See [here](https://github.com/TensorSpeech/TensorFlowTTS/tree/master/examples/parallel_wavegan)
- 2020/08/20 Add C++ inference code. Thank [@ZDisket](https://github.com/ZDisket). See [here](https://github.com/TensorSpeech/TensorFlowTTS/tree/master/examples/cppwin)
- 2020/08/18 Update [new base processor](https://github.com/TensorSpeech/TensorFlowTTS/blob/master/tensorflow_tts/processor/base_processor.py). Add [AutoProcessor](https://github.com/TensorSpeech/TensorFlowTTS/blob/master/tensorflow_tts/inference/auto_processor.py) and [pretrained processor](https://github.com/TensorSpeech/TensorFlowTTS/blob/master/tensorflow_tts/processor/pretrained/) json file
- 2020/08/14 Support Chinese TTS. Pls see the [colab](https://colab.research.google.com/drive/1YpSHRBRPBI7cnTkQn1UcVTWEQVbsUm1S?usp=sharing). Thank [@azraelkuan](https://github.com/azraelkuan)
- 2020/08/05 Support Korean TTS. Pls see the [colab](https://colab.research.google.com/drive/1ybWwOS5tipgPFttNulp77P6DAB5MtiuN?usp=sharing). Thank [@crux153](https://github.com/crux153)
- 2020/07/17 Support MultiGPU for all Trainer
- 2020/07/05 Support Convert Tacotron-2, FastSpeech to Tflite. Pls see the [colab](https://colab.research.google.com/drive/1HudLLpT9CQdh2k04c06bHUwLubhGTWxA?usp=sharing). Thank @jaeyoo from the TFlite team for his support
- 2020/06/20 [FastSpeech2](https://arxiv.org/abs/2006.04558) implementation with Tensorflow is supported.
- 2020/06/07 [Multi-band MelGAN (MB MelGAN)](https://github.com/tensorspeech/TensorFlowTTS/blob/master/examples/multiband_melgan/) implementation with Tensorflow is supported


## Features
- High performance on Speech Synthesis.
- Be able to fine-tune on other languages.
- Fast, Scalable, and Reliable.
- Suitable for deployment.
- Easy to implement a new model, based-on abstract class.
- Mixed precision to speed-up training if possible.
- Support Single/Multi GPU gradient Accumulate.
- Support both Single/Multi GPU in base trainer class.
- TFlite conversion for all supported models.
- Android example.
- Support many languages (currently, we support Chinese, Korean, English, French and German)
- Support C++ inference.
- Support Convert weight for some models from PyTorch to TensorFlow to accelerate speed.

## Requirements
This repository is tested on Ubuntu 18.04 with:

- Python 3.7+
- Cuda 10.1
- CuDNN 7.6.5
- Tensorflow 2.2/2.3/2.4/2.5/2.6
- [Tensorflow Addons](https://github.com/tensorflow/addons) >= 0.10.0

Different Tensorflow version should be working but not tested yet. This repo will try to work with the latest stable TensorFlow version. **We recommend you install TensorFlow 2.6.0 to training in case you want to use MultiGPU.**

## Installation
### With pip
```bash
$ pip install TensorFlowTTS
```
### From source
Examples are included in the repository but are not shipped with the framework. Therefore, to run the latest version of examples, you need to install the source below.
```bash
$ git clone https://github.com/TensorSpeech/TensorFlowTTS.git
$ cd TensorFlowTTS
$ pip install .
```
If you want to upgrade the repository and its dependencies:
```bash
$ git pull
$ pip install --upgrade .
```

# Supported Model architectures
TensorFlowTTS currently  provides the following architectures:

1. **MelGAN** released with the paper [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711) by Kundan Kumar, Rithesh Kumar, Thibault de Boissiere, Lucas Gestin, Wei Zhen Teoh, Jose Sotelo, Alexandre de Brebisson, Yoshua Bengio, Aaron Courville.
2. **Tacotron-2** released with the paper [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) by Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, Yonghui Wu.
3. **FastSpeech** released with the paper [FastSpeech: Fast, Robust, and Controllable Text to Speech](https://arxiv.org/abs/1905.09263) by Yi Ren, Yangjun Ruan, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu.
4. **Multi-band MelGAN** released with the paper [Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech](https://arxiv.org/abs/2005.05106) by Geng Yang, Shan Yang, Kai Liu, Peng Fang, Wei Chen, Lei Xie.
5. **FastSpeech2** released with the paper [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558) by Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu.
6. **Parallel WaveGAN** released with the paper [Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://arxiv.org/abs/1910.11480) by Ryuichi Yamamoto, Eunwoo Song, Jae-Min Kim.
7. **HiFi-GAN** released with the paper [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646) by Jungil Kong, Jaehyeon Kim, Jaekyoung Bae.

We are also implementing some techniques to improve quality and convergence speed from the following papers:

2. **Guided Attention Loss** released with the paper [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention
](https://arxiv.org/abs/1710.08969) by Hideyuki Tachibana, Katsuya Uenoyama, Shunsuke Aihara.


# Audio Samples
Here in an audio samples on valid set. [tacotron-2](https://drive.google.com/open?id=1kaPXRdLg9gZrll9KtvH3-feOBMM8sn3_), [fastspeech](https://drive.google.com/open?id=1f69ujszFeGnIy7PMwc8AkUckhIaT2OD0), [melgan](https://drive.google.com/open?id=1mBwGVchwtNkgFsURl7g4nMiqx4gquAC2), [melgan.stft](https://drive.google.com/open?id=1xUkDjbciupEkM3N4obiJAYySTo6J9z6b), [fastspeech2](https://drive.google.com/drive/u/1/folders/1NG7oOfNuXSh7WyAoM1hI8P5BxDALY_mU), [multiband_melgan](https://drive.google.com/drive/folders/1DCV3sa6VTyoJzZmKATYvYVDUAFXlQ_Zp)

# Tutorial End-to-End

## Prepare Dataset

Prepare a dataset in the following format:
```
|- [NAME_DATASET]/
|   |- metadata.csv
|   |- wavs/
|       |- file1.wav
|       |- ...
```

Where `metadata.csv` has the following format: `id|transcription`. This is a ljspeech-like format; you can ignore preprocessing steps if you have other format datasets.

Note that `NAME_DATASET` should be `[ljspeech/kss/baker/libritts/synpaflex]` for example.

## Preprocessing

The preprocessing has two steps:

1. Preprocess audio features
    - Convert characters to IDs
    - Compute mel spectrograms
    - Normalize mel spectrograms to [-1, 1] range
    - Split the dataset into train and validation
    - Compute the mean and standard deviation of multiple features from the **training** split
2. Standardize mel spectrogram based on computed statistics

To reproduce the steps above:
```
tensorflow-tts-preprocess --rootdir ./[ljspeech/kss/baker/libritts/thorsten/synpaflex] --outdir ./dump_[ljspeech/kss/baker/libritts/thorsten/synpaflex] --config preprocess/[ljspeech/kss/baker/thorsten/synpaflex]_preprocess.yaml --dataset [ljspeech/kss/baker/libritts/thorsten/synpaflex]
tensorflow-tts-normalize --rootdir ./dump_[ljspeech/kss/baker/libritts/thorsten/synpaflex] --outdir ./dump_[ljspeech/kss/baker/libritts/thorsten/synpaflex] --config preprocess/[ljspeech/kss/baker/libritts/thorsten/synpaflex]_preprocess.yaml --dataset [ljspeech/kss/baker/libritts/thorsten/synpaflex]
```

Right now we only support [`ljspeech`](https://keithito.com/LJ-Speech-Dataset/), [`kss`](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset), [`baker`](https://weixinxcxdb.oss-cn-beijing.aliyuncs.com/gwYinPinKu/BZNSYP.rar), [`libritts`](http://www.openslr.org/60/), [`thorsten`](https://github.com/thorstenMueller/deep-learning-german-tts) and
[`synpaflex`](https://www.ortolang.fr/market/corpora/synpaflex-corpus/) for dataset argument. In the future, we intend to support more datasets.

**Note**: To run `libritts` preprocessing, please first read the instruction in [examples/fastspeech2_libritts](https://github.com/TensorSpeech/TensorFlowTTS/tree/master/examples/fastspeech2_libritts). We need to reformat it first before run preprocessing.

**Note**: To run `synpaflex` preprocessing, please first run the notebook [notebooks/prepare_synpaflex.ipynb](https://github.com/TensorSpeech/TensorFlowTTS/tree/master/notebooks/prepare_synpaflex.ipynb). We need to reformat it first before run preprocessing.

After preprocessing, the structure of the project folder should be:
```
|- [NAME_DATASET]/
|   |- metadata.csv
|   |- wav/
|       |- file1.wav
|       |- ...
|- dump_[ljspeech/kss/baker/libritts/thorsten]/
|   |- train/
|       |- ids/
|           |- LJ001-0001-ids.npy
|           |- ...
|       |- raw-feats/
|           |- LJ001-0001-raw-feats.npy
|           |- ...
|       |- raw-f0/
|           |- LJ001-0001-raw-f0.npy
|           |- ...
|       |- raw-energies/
|           |- LJ001-0001-raw-energy.npy
|           |- ...
|       |- norm-feats/
|           |- LJ001-0001-norm-feats.npy
|           |- ...
|       |- wavs/
|           |- LJ001-0001-wave.npy
|           |- ...
|   |- valid/
|       |- ids/
|           |- LJ001-0009-ids.npy
|           |- ...
|       |- raw-feats/
|           |- LJ001-0009-raw-feats.npy
|           |- ...
|       |- raw-f0/
|           |- LJ001-0001-raw-f0.npy
|           |- ...
|       |- raw-energies/
|           |- LJ001-0001-raw-energy.npy
|           |- ...
|       |- norm-feats/
|           |- LJ001-0009-norm-feats.npy
|           |- ...
|       |- wavs/
|           |- LJ001-0009-wave.npy
|           |- ...
|   |- stats.npy
|   |- stats_f0.npy
|   |- stats_energy.npy
|   |- train_utt_ids.npy
|   |- valid_utt_ids.npy
|- examples/
|   |- melgan/
|   |- fastspeech/
|   |- tacotron2/
|   ...
```

- `stats.npy` contains the mean and std from the training split mel spectrograms
- `stats_energy.npy` contains the mean and std of energy values from the training split
- `stats_f0.npy` contains the mean and std of F0 values in the training split
- `train_utt_ids.npy` / `valid_utt_ids.npy` contains training and validation utterances IDs respectively

We use suffix (`ids`, `raw-feats`, `raw-energy`, `raw-f0`, `norm-feats`, and `wave`) for each input type.


**IMPORTANT NOTES**:
- This preprocessing step is based on [ESPnet](https://github.com/espnet/espnet) so you can combine all models here with other models from ESPnet repository.
- Regardless of how your dataset is formatted, the final structure of the `dump` folder **SHOULD** follow the above structure to be able to use the training script, or you can modify it by yourself 😄.

## Training models

To know how to train model from scratch or fine-tune with other datasets/languages, please see detail at example directory.

- For Tacotron-2 tutorial, pls see [examples/tacotron2](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/tacotron2)
- For FastSpeech tutorial, pls see [examples/fastspeech](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/fastspeech)
- For FastSpeech2 tutorial, pls see [examples/fastspeech2](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/fastspeech2)
- For FastSpeech2 + MFA tutorial, pls see [examples/fastspeech2_libritts](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/fastspeech2_libritts)
- For MelGAN tutorial, pls see [examples/melgan](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/melgan)
- For MelGAN + STFT Loss tutorial, pls see [examples/melgan.stft](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/melgan.stft)
- For Multiband-MelGAN tutorial, pls see [examples/multiband_melgan](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/multiband_melgan)
- For Parallel WaveGAN tutorial, pls see [examples/parallel_wavegan](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/parallel_wavegan)
- For Multiband-MelGAN Generator + HiFi-GAN tutorial, pls see [examples/multiband_melgan_hf](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/multiband_melgan_hf)
- For HiFi-GAN tutorial, pls see [examples/hifigan](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/hifigan)
# Abstract Class Explaination

## Abstract DataLoader Tensorflow-based dataset

A detail implementation of abstract dataset class from [tensorflow_tts/dataset/abstract_dataset](https://github.com/tensorspeech/TensorFlowTTS/blob/master/tensorflow_tts/datasets/abstract_dataset.py). There are some functions you need overide and understand:

1. **get_args**: This function return argumentation for **generator** class, normally is utt_ids.
2. **generator**: This function have an inputs from **get_args** function and return a inputs for models. **Note that we return a dictionary for all generator functions with the keys that exactly match with the model's parameters because base_trainer will use model(\*\*batch) to do forward step.**
3. **get_output_dtypes**: This function need return dtypes for each element from **generator** function.
4. **get_len_dataset**: Return len of datasets, normaly is len(utt_ids).

**IMPORTANT NOTES**:

- A pipeline of creating dataset should be: cache -> shuffle -> map_fn -> get_batch -> prefetch.
- If you do shuffle before cache, the dataset won't shuffle when it re-iterate over datasets.
- You should apply map_fn to make each element return from **generator** function have the same length before getting batch and feed it into a model.

Some examples to use this **abstract_dataset** are [tacotron_dataset.py](https://github.com/tensorspeech/TensorFlowTTS/blob/master/examples/tacotron2/tacotron_dataset.py), [fastspeech_dataset.py](https://github.com/tensorspeech/TensorFlowTTS/blob/master/examples/fastspeech/fastspeech_dataset.py), [melgan_dataset.py](https://github.com/tensorspeech/TensorFlowTTS/blob/master/examples/melgan/audio_mel_dataset.py), [fastspeech2_dataset.py](https://github.com/TensorSpeech/TensorFlowTTS/blob/master/examples/fastspeech2/fastspeech2_dataset.py)


## Abstract Trainer Class

A detail implementation of base_trainer from [tensorflow_tts/trainer/base_trainer.py](https://github.com/tensorspeech/TensorFlowTTS/blob/master/tensorflow_tts/trainers/base_trainer.py). It include [Seq2SeqBasedTrainer](https://github.com/tensorspeech/TensorFlowTTS/blob/master/tensorflow_tts/trainers/base_trainer.py#L265) and [GanBasedTrainer](https://github.com/tensorspeech/TensorFlowTTS/blob/master/tensorflow_tts/trainers/base_trainer.py#L149) inherit from [BasedTrainer](https://github.com/tensorspeech/TensorFlowTTS/blob/master/tensorflow_tts/trainers/base_trainer.py#L16). All trainer support both single/multi GPU. There a some functions you **MUST** overide when implement new_trainer:

- **compile**: This function aim to define a models, and losses.
- **generate_and_save_intermediate_result**: This function will save intermediate result such as: plot alignment, save audio generated, plot mel-spectrogram ...
- **compute_per_example_losses**: This function will compute per_example_loss for model, note that all element of the loss **MUST** has shape [batch_size].

All models on this repo are trained based-on **GanBasedTrainer** (see [train_melgan.py](https://github.com/tensorspeech/TensorFlowTTS/blob/master/examples/melgan/train_melgan.py), [train_melgan_stft.py](https://github.com/tensorspeech/TensorFlowTTS/blob/master/examples/melgan.stft/train_melgan_stft.py), [train_multiband_melgan.py](https://github.com/tensorspeech/TensorFlowTTS/blob/master/examples/multiband_melgan/train_multiband_melgan.py)) and **Seq2SeqBasedTrainer** (see [train_tacotron2.py](https://github.com/tensorspeech/TensorFlowTTS/blob/master/examples/tacotron2/train_tacotron2.py), [train_fastspeech.py](https://github.com/tensorspeech/TensorFlowTTS/blob/master/examples/fastspeech/train_fastspeech.py)).

# End-to-End Examples
You can know how to inference each model at [notebooks](https://github.com/tensorspeech/TensorFlowTTS/tree/master/notebooks) or see a [colab](https://colab.research.google.com/drive/1akxtrLZHKuMiQup00tzO2olCaN-y3KiD?usp=sharing) (for English), [colab](https://colab.research.google.com/drive/1ybWwOS5tipgPFttNulp77P6DAB5MtiuN?usp=sharing) (for Korean), [colab](https://colab.research.google.com/drive/1YpSHRBRPBI7cnTkQn1UcVTWEQVbsUm1S?usp=sharing) (for Chinese), [colab](https://colab.research.google.com/drive/1jd3u46g-fGQw0rre8fIwWM9heJvrV1c0?usp=sharing) (for French), [colab](https://colab.research.google.com/drive/1W0nSFpsz32M0OcIkY9uMOiGrLTPKVhTy?usp=sharing) (for German). Here is an example code for end2end inference with fastspeech2 and multi-band melgan. We uploaded all our pretrained in [HuggingFace Hub](https://huggingface.co/tensorspeech).

```python
import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

# initialize fastspeech2 model.
fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")


# initialize mb_melgan model
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")


# inference
processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")

input_ids = processor.text_to_sequence("Recent research at Harvard has shown meditating for as little as 8 weeks, can actually increase the grey matter in the parts of the brain responsible for emotional regulation, and learning.")
# fastspeech inference

mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
    energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
)

# melgan inference
audio_before = mb_melgan.inference(mel_before)[0, :, 0]
audio_after = mb_melgan.inference(mel_after)[0, :, 0]

# save to file
sf.write('./audio_before.wav', audio_before, 22050, "PCM_16")
sf.write('./audio_after.wav', audio_after, 22050, "PCM_16")
```

# Contact
- [Minh Nguyen Quan Anh](https://github.com/tensorspeech): nguyenquananhminh@gmail.com
- [erogol](https://github.com/erogol): erengolge@gmail.com
- [Kuan Chen](https://github.com/azraelkuan): azraelkuan@gmail.com
- [Dawid Kobus](https://github.com/machineko): machineko@protonmail.com
- [Takuya Ebata](https://github.com/MokkeMeguru): meguru.mokke@gmail.com
- [Trinh Le Quang](https://github.com/l4zyf9x): trinhle.cse@gmail.com
- [Yunchao He](https://github.com/candlewill): yunchaohe@gmail.com
- [Alejandro Miguel Velasquez](https://github.com/ZDisket): xml506ok@gmail.com

# License
All models here are licensed under the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

# Acknowledgement
We want to thank [Tomoki Hayashi](https://github.com/kan-bayashi), who discussed with us much about Melgan, Multi-band melgan, Fastspeech, and Tacotron. This framework based-on his great open-source [ParallelWaveGan](https://github.com/kan-bayashi/ParallelWaveGAN) project.
