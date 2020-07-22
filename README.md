<h2 align="center">
<p> :yum: TensorflowTTS
<p align="center">
    <a href="https://github.com/dathudeptrai/TensorflowTTS/actions">
        <img alt="Build" src="https://github.com/dathudeptrai/TensorflowTTS/workflows/CI/badge.svg?branch=master">
    </a>
    <a href="https://github.com/dathudeptrai/TensorflowTTS/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/dathudeptrai/TensorflowTTS?color=red">
    </a>
    <a href="https://colab.research.google.com/drive/1akxtrLZHKuMiQup00tzO2olCaN-y3KiD?usp=sharing">
        <img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
</p>
</h2>
<h2 align="center">
<p>Real-Time State-of-the-art Speech Synthesis for Tensorflow 2
</h2>

:zany_face: TensorflowTTS provides real-time state-of-the-art speech synthesis architectures such as Tacotron-2, Melgan, Multiband-Melgan, FastSpeech, FastSpeech2 based-on TensorFlow 2. With Tensorflow 2, we can speed-up training/inference progress, optimizer further by using [fake-quantize aware](https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide) and [pruning](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras), make TTS models can be run faster than real-time and be able to deploy on mobile devices or embedded systems. 

## What's new
- 2020/07/17 **(NEW!)** Support MultiGPU for all Trainer.
- 2020/07/05 **(New!)** Support Convert Tacotron-2, FastSpeech to Tflite. Pls see the [colab](https://colab.research.google.com/drive/1HudLLpT9CQdh2k04c06bHUwLubhGTWxA?usp=sharing). Thank @jaeyoo from TFlite team for his support.
- 2020/06/20 [FastSpeech2](https://arxiv.org/abs/2006.04558) implementation with Tensorflow is supported.
- 2020/06/07 [Multi-band MelGAN (MB MelGAN)](https://github.com/dathudeptrai/TensorflowTTS/blob/master/examples/multiband_melgan/) implementation with Tensorflow is supported. 


## Features
- High performance on Speech Synthesis.
- Be able to fine-tune on other languages.
- Fast, Scalable and Reliable.
- Suitable for deployment.
- Easy to implement new model based-on abtract class.
- Mixed precision to speed-up training if posible.
- Support both Single/Multi GPU in base trainer class.
- TFlite conversion for all supported model.

## Requirements
This repository is tested on Ubuntu 18.04 with:

- Python 3.6+
- Cuda 10.1
- CuDNN 7.6.5
- Tensorflow 2.2
- [Tensorflow Addons](https://github.com/tensorflow/addons) 0.10.0

Different Tensorflow version should be working but not tested yet. This repo will try to work with latest stable tensorflow version. **We recommend you install tensorflow 2.3.0 to training in case you want to use MultiGPU.**

## Installation
### With pip
```bash
$ pip install TensorflowTTS
```
### From source 
Examples are included in the repository but are not shipped with the framework. Therefore, in order to run the latest verion of examples, you need install from source following bellow.
```bash
$ git clone https://github.com/TensorSpeech/TensorflowTTS.git
$ cd TensorflowTTS
$ pip install  .
```
If you want upgrade the repository and its dependencies:
```bash
$ git pull
$ pip install --upgrade .
```

# Supported Model achitectures
TensorflowTTS currently  provides the following architectures:

1. **MelGAN** released with the paper [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711) by Kundan Kumar, Rithesh Kumar, Thibault de Boissiere, Lucas Gestin, Wei Zhen Teoh, Jose Sotelo, Alexandre de Brebisson, Yoshua Bengio, Aaron Courville.
2. **Tacotron-2** released with the paper [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) by Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, Yonghui Wu.
3. **FastSpeech** released with the paper [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263) by Yi Ren, Yangjun Ruan, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu.
4. **Multi-band MelGAN** released with the paper [Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech](https://arxiv.org/abs/2005.05106) by Geng Yang, Shan Yang, Kai Liu, Peng Fang, Wei Chen, Lei Xie.
5. **FastSpeech2** released with the paper [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558) by Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu.

We are also implement some techniques to improve quality and convergence speed from following papers:

1. **Multi Resolution STFT Loss** released with the paper [Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://arxiv.org/abs/1910.11480) by Ryuichi Yamamoto, Eunwoo Song, Jae-Min Kim.
2. **Guided Attention Loss** released with the paper [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention
](https://arxiv.org/abs/1710.08969) by Hideyuki Tachibana, Katsuya Uenoyama, Shunsuke Aihara.


# Audio Samples
Here in an audio samples on valid set. [tacotron-2](https://drive.google.com/open?id=1kaPXRdLg9gZrll9KtvH3-feOBMM8sn3_), [fastspeech](https://drive.google.com/open?id=1f69ujszFeGnIy7PMwc8AkUckhIaT2OD0), [melgan](https://drive.google.com/open?id=1mBwGVchwtNkgFsURl7g4nMiqx4gquAC2), [melgan.stft](https://drive.google.com/open?id=1xUkDjbciupEkM3N4obiJAYySTo6J9z6b), [fastspeech2](https://drive.google.com/drive/u/1/folders/1NG7oOfNuXSh7WyAoM1hI8P5BxDALY_mU), [multiband_melgan](https://drive.google.com/drive/folders/1DCV3sa6VTyoJzZmKATYvYVDUAFXlQ_Zp)

# Tutorial End-to-End

## Prepare Dataset

Prepare a dataset in the following format:
```
|- datasets/
|   |- metadata.csv
|   |- wav/
|       |- file1.wav
|       |- ...
```

where metadata.csv has the following format: id|transcription. This is a ljspeech-like format, you can ignore preprocessing step if you have other format dataset.

## Preprocessing

The preprocessing have three steps:

1. Convert charactor to ids, calculate pre-normalize melspectrogram, normalize audio to [-1, 1], split dataset into train and valid part.
2. Computer mean/var of melspectrogram over **training** part.
3. Normalize melspectrogram based-on mean/var of training dataset.

This is a command line to do three steps above:

```
tensorflow-tts-preprocess --rootdir ./datasets/ --outdir ./dump/ --conf preprocess/ljspeech_preprocess.yaml
tensorflow-tts-compute-statistics --rootdir ./dump/train/ --outdir ./dump --config preprocess/ljspeech_preprocess.yaml
tensorflow-tts-normalize --rootdir ./dump --outdir ./dump --stats ./dump/stats.npy --config preprocess/ljspeech_preprocess.yaml

```

After preprocessing, a structure of project will become:
```
|- datasets/
|   |- metadata.csv
|   |- wav/
|       |- file1.wav
|       |- ...
|- dump/
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
|   |- stats.npy/ 
|   |- stats_f0.npy/ 
|   |- stats_energy.npy/ 
|   |- train_utt_ids.npy
|   |- valid_utt_ids.npy
|- examples/
|   |- melgan/
|   |- fastspeech/
|   |- tacotron2/
|   ...
```

Where stats.npy contains mean/var of train melspectrogram (we can use mean/var to de-normalization to get melspectrogram raw), stats_energy.npy is a min/max value of energy values over **Training** dataset, stats_f0 is a min/max value of F0 values, train_utt_ids/valid_utt_ids contains training and valid utt ids respectively. We use suffix (ids, raw-feats, norm-feats, wave) for each type of input.

**IMPORTANT NOTES**:
- This preprocessing step based-on [ESP-NET](https://github.com/espnet/espnet) so you can combine all models here with other models from espnet repo.

## Training models

To know how to training model from scratch or fine-tune with other datasets/languages, pls see detail at example directory.

- For Tacotron-2 tutorial, pls see [example/tacotron2](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/tacotron2)
- For FastSpeech tutorial, pls see [example/fastspeech](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/fastspeech)
- For FastSpeech2 tutorial, pls see [example/fastspeech2](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/fastspeech2)
- For MelGAN tutorial, pls see [example/melgan](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/melgan)
- For MelGAN + STFT Loss tutorial, pls see [example/melgan.stft](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/melgan.stft)
- For Multiband-MelGAN tutorial, pls see [example/multiband_melgan](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/multiband_melgan)
# Abstract Class Explaination

## Abstract DataLoader Tensorflow-based dataset

A detail implementation of abstract dataset class from [tensorflow_tts/dataset/abstract_dataset](https://github.com/dathudeptrai/TensorflowTTS/blob/master/tensorflow_tts/datasets/abstract_dataset.py). There are some functions you need overide and understand:

1. **get_args**: This function return argumentation for **generator** class, normally is utt_ids.
2. **generator**: This funtion have an inputs from **get_args** function and return a inputs for models. **Note that we return dictionary for all generator function with they keys exactly match with the parameter of the model because base_trainer will use model(\*\*batch) to do forward step.**
3. **get_output_dtypes**: This function need return dtypes for each element from **generator** function.
4. **get_len_dataset**: Return len of datasets, normaly is len(utt_ids).

**IMPORTANT NOTES**:

- A pipeline of creating dataset should be: cache -> shuffle -> map_fn -> get_batch -> prefetch.
- If you do shuffle before cache, the dataset won't shuffle when it re-iterate over datasets.
- You should apply map_fn to make each elements return from **generator** function have a same length before get batch and feed it into a model.

Some examples to use this **abstract_dataset** are [tacotron_dataset.py](https://github.com/dathudeptrai/TensorflowTTS/blob/master/examples/tacotron2/tacotron_dataset.py), [fastspeech_dataset.py](https://github.com/dathudeptrai/TensorflowTTS/blob/master/examples/fastspeech/fastspeech_dataset.py), [melgan_dataset.py](https://github.com/dathudeptrai/TensorflowTTS/blob/master/examples/melgan/audio_mel_dataset.py), [fastspeech2_dataset.py](https://github.com/TensorSpeech/TensorflowTTS/blob/master/examples/fastspeech2/fastspeech2_dataset.py)


## Abstract Trainer Class

A detail implementation of base_trainer from [tensorflow_tts/trainer/base_trainer.py](https://github.com/dathudeptrai/TensorflowTTS/blob/master/tensorflow_tts/trainers/base_trainer.py). It include [Seq2SeqBasedTrainer](https://github.com/dathudeptrai/TensorflowTTS/blob/master/tensorflow_tts/trainers/base_trainer.py#L265) and [GanBasedTrainer](https://github.com/dathudeptrai/TensorflowTTS/blob/master/tensorflow_tts/trainers/base_trainer.py#L149) inherit from [BasedTrainer](https://github.com/dathudeptrai/TensorflowTTS/blob/master/tensorflow_tts/trainers/base_trainer.py#L16). All trainer support both single/multi GPU. There a some functions you **MUST** overide when implement new_trainer:

- **compile**: This function aim to define a models, and losses.
- **generate_and_save_intermediate_result**: This function will save intermediate result such as: plot alignment, save audio generated, plot mel-spectrogram ...
- **compute_per_example_losses**: This function will compute per_example_loss for model, note that all element of the loss **MUST** has shape [batch_size].

All models on this repo are trained based-on **GanBasedTrainer** (see [train_melgan.py](https://github.com/dathudeptrai/TensorflowTTS/blob/master/examples/melgan/train_melgan.py), [train_melgan_stft.py](https://github.com/dathudeptrai/TensorflowTTS/blob/master/examples/melgan.stft/train_melgan_stft.py), [train_multiband_melgan.py](https://github.com/dathudeptrai/TensorflowTTS/blob/master/examples/multiband_melgan/train_multiband_melgan.py)) and **Seq2SeqBasedTrainer** (see [train_tacotron2.py](https://github.com/dathudeptrai/TensorflowTTS/blob/master/examples/tacotron2/train_tacotron2.py), [train_fastspeech.py](https://github.com/dathudeptrai/TensorflowTTS/blob/master/examples/fastspeech/train_fastspeech.py)). 

# End-to-End Examples
You can know how to inference each model at [notebooks](https://github.com/dathudeptrai/TensorflowTTS/tree/master/notebooks) or see a [colab](https://colab.research.google.com/drive/1akxtrLZHKuMiQup00tzO2olCaN-y3KiD?usp=sharing). Here is an example code for end2end inference with fastspeech and melgan.

```python
import numpy as np
import soundfile as sf
import yaml

import tensorflow as tf

from tensorflow_tts.processor import LJSpeechProcessor

from tensorflow_tts.configs import FastSpeechConfig
from tensorflow_tts.configs import MelGANGeneratorConfig

from tensorflow_tts.models import TFFastSpeech
from tensorflow_tts.models import TFMelGANGenerator

# initialize fastspeech model.
with open('./examples/fastspeech/conf/fastspeech.v1.yaml') as f:
    fs_config = yaml.load(f, Loader=yaml.Loader)
fs_config = FastSpeechConfig(**fs_config["fastspeech_params"])
fastspeech = TFFastSpeech(config=fs_config, name="fastspeech")
fastspeech._build()
fastspeech.load_weights("./examples/fastspeech/pretrained/model-195000.h5")

# initialize melgan model
with open('./examples/melgan/conf/melgan.v1.yaml') as f:
    melgan_config = yaml.load(f, Loader=yaml.Loader)
melgan_config = MelGANGeneratorConfig(**melgan_config["generator_params"])
melgan = TFMelGANGenerator(config=melgan_config, name='melgan_generator')
melgan._build()
melgan.load_weights("./examples/melgan/pretrained/generator-1500000.h5")


# inference
processor = LJSpeechProcessor(None, cleaner_names="english_cleaners")

ids = processor.text_to_sequence("Recent research at Harvard has shown meditating for as little as 8 weeks, can actually increase the grey matter in the parts of the brain responsible for emotional regulation, and learning.")
ids = tf.expand_dims(ids, 0)
# fastspeech inference

masked_mel_before, masked_mel_after, duration_outputs = fastspeech.inference(
    ids,
    speaker_ids=tf.zeros(shape=[tf.shape(ids)[0]]),
    speed_ratios=tf.constant([1.0], dtype=tf.float32)
)

# melgan inference
audio_before = melgan(masked_mel_before)[0, :, 0]
audio_after = melgan(masked_mel_after)[0, :, 0]

# save to file
sf.write('./audio_before.wav', audio_before, 22050, "PCM_16")
sf.write('./audio_after.wav', audio_after, 22050, "PCM_16")
```

# Contact
[Minh Nguyen Quan Anh](https://github.com/dathudeptrai): nguyenquananhminh@gmail.com, [erogol](https://github.com/erogol): erengolge@gmail.com, [Kuan Chen](https://github.com/azraelkuan): azraelkuan@gmail.com, [Takuya Ebata](https://github.com/MokkeMeguru): meguru.mokke@gmail.com, [Trinh Le Quang](https://github.com/l4zyf9x): trinhle.cse@gmail.com

# License
Overrall, Almost models here are licensed under the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) for all countries in the world, except in **Viet Nam** this framework cannot be used for production in any way without permission from TensorflowTTS's Authors. There is an exception, Tacotron-2 can be used with any perpose. So, if you are VietNamese and want to use this framework for production, you **Must** contact our in andvance.

# Acknowledgement
We would like to thank [Tomoki Hayashi](https://github.com/kan-bayashi), who discussed with our much about Melgan, Multi-band melgan, Fastspeech and Tacotron. This framework based-on his great open-source [ParallelWaveGan](https://github.com/kan-bayashi/ParallelWaveGAN) project. 
