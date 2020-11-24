# HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis
Based on the script [`train_hifigan.py`](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/hifigan/train_hifigan.py).

## Training HiFi-GAN from scratch with LJSpeech dataset.
This example code show you how to train MelGAN from scratch with Tensorflow 2 based on custom training loop and tf.function. The data used for this example is LJSpeech, you can download the dataset at  [link](https://keithito.com/LJ-Speech-Dataset/).

### Step 1: Create Tensorflow based Dataloader (tf.dataset)
First, you need define data loader based on AbstractDataset class (see [`abstract_dataset.py`](https://github.com/tensorspeech/TensorFlowTTS/tree/master/tensorflow_tts/datasets/abstract_dataset.py)). On this example, a dataloader read dataset from path. I use suffix to classify what file is a audio and mel-spectrogram (see [`audio_mel_dataset.py`](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/melgan/audio_mel_dataset.py)). If you already have preprocessed version of your target dataset, you don't need to use this example dataloader, you just need refer my dataloader and modify **generator function** to adapt with your case. Normally, a generator function should return [audio, mel].

### Step 2: Training from scratch
After you re-define your dataloader, pls modify an input arguments, train_dataset and valid_dataset from [`train_hifigan.py`](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/hifigan/train_hifigan.py). Here is an example command line to training HiFi-GAN from scratch:

First, you need training generator with only stft loss: 

```bash
CUDA_VISIBLE_DEVICES=0 python examples/hifigan/train_hifigan.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./examples/hifigan/exp/train.hifigan.v1/ \
  --config ./examples/hifigan/conf/hifigan.v1.yaml \
  --use-norm 1
  --generator_mixed_precision 1 \
  --resume ""
```

Then resume and start training generator + discriminator:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/hifigan/train_hifigan.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./examples/hifigan/exp/train.hifigan.v1/ \
  --config ./examples/hifigan/conf/hifigan.v1.yaml \
  --use-norm 1
  --resume ./examples/hifigan/exp/train.hifigan.v1/checkpoints/ckpt-100000
```

IF you want to use MultiGPU to training you can replace `CUDA_VISIBLE_DEVICES=0` by `CUDA_VISIBLE_DEVICES=0,1,2,3` for example. You also need to tune the `batch_size` for each GPU (in config file) by yourself to maximize the performance. Note that MultiGPU now support for Training but not yet support for Decode.

In case you want to resume the training progress, please following below example command line:

```bash
--resume ./examples/hifigan/exp/train.hifigan.v1/checkpoints/ckpt-100000
```

If you want to finetune a model, use `--pretrained` like this with the filename of the generator
```bash
--pretrained ptgenerator.h5
```

**IMPORTANT NOTES**:

- When training generator only, we enable mixed precision to speed-up training progress.
- We don't apply mixed precision when training both generator and discriminator. (Discriminator include group-convolution, which cause discriminator slower when enable mixed precision).
- 100k here is a *discriminator_train_start_steps* parameters from [hifigan.v1.yaml](https://github.com/tensorspeech/TensorflowTTS/tree/master/examples/hifigan/conf/hifigan.v1.yaml)


## Reference

1. https://github.com/descriptinc/melgan-neurips
2. https://github.com/kan-bayashi/ParallelWaveGAN
3. https://github.com/tensorflow/addons
4. [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
5. [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711)
6. [Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://arxiv.org/abs/1910.11480)