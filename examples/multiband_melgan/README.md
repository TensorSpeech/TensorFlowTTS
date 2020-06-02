# Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech
Based on the script [`train_multiband_melgan.py`](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/multiband_melgan/train_multiband_melgan.py).

## Training Multi-band MelGAN from scratch with LJSpeech dataset.
This example code show you how to train MelGAN from scratch with Tensorflow 2 based on custom training loop and tf.function. The data used for this example is LJSpeech, you can download the dataset at  [link](https://keithito.com/LJ-Speech-Dataset/).

### Step 1: Create Tensorflow based Dataloader (tf.dataset)
Please see detail at [examples/melgan/](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/melgan#step-1-create-tensorflow-based-dataloader-tfdataset)

### Step 2: Training from scratch
After you re-define your dataloader, pls modify an input arguments, train_dataset and valid_dataset from [`train_multiband_melgan.py`](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/multiband_melgan/train_multiband_melgan.py). Here is an example command line to training melgan-stft from scratch:

First, you need training generator with only stft loss: 

```bash
CUDA_VISIBLE_DEVICES=0 python examples/multiband_melgan/train_multiband_melgan.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./examples/multiband_melgan/exp/train.multiband_melgan.v1/ \
  --config ./examples/multiband_melgan/conf/multiband_melgan.v1.yaml \
  --use-norm 1
  --generator_mixed_precision 1 \
  --resume ""
```

Then resume and start training generator + discriminator:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/multiband_melgan/train_multiband_melgan.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./examples/multiband_melgan/exp/train.multiband_melgan.v1/ \
  --config ./examples/multiband_melgan/conf/multiband_melgan.v1.yaml \
  --use-norm 1
  --resume ./examples/multiband_melgan/exp/train.multiband_melgan.v1/checkpoints/ckpt-200000
```

## Finetune MelGAN STFT with ljspeech pretrained on other languages
Just load pretrained model and training from scratch with other languages. **DO NOT FORGET** re-preprocessing on your dataset if needed. A hop_size should be 256 if you want to use our pretrained.

## Learning Curves
Comming soon...


## Pretrained Models and Audio samples
Comming soon...

## Reference

1. https://github.com/descriptinc/melgan-neurips
2. https://github.com/kan-bayashi/ParallelWaveGAN
3. [Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://arxiv.org/abs/1910.11480)
4. [Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech](https://arxiv.org/abs/2005.05106)