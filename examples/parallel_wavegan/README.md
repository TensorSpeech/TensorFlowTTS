# Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram
Based on the script [`train_parallel_wavegan.py`](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/parallel_wavegan/train_parallel_wavegan.py).


## Convert pretrained weight from Pytorch Parallel WaveGAN to TensorFlow Parallel WaveGAN to Accelerate Inference Speed and Deployability 

We recommand users use pytorch Parallel WaveGAN from [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) to training for convenient and very stable. After finish training, you can convert the pytorch's weight to this tensorflow pwgan version to accelerate inference speech and enhance deployability. You can use the pretrained weight from [here](https://github.com/kan-bayashi/ParallelWaveGAN#results) then use [convert_pwgan_from_pytorch_to_tensorflow](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/parallel_wavegan/convert_pwgan_from_pytorch_to_tensorflow.ipynp) notebook to convert it. Note that the pwgan pretrained weight from pytorch repo can be use as vocoder with our text2mel model because they uses the same preprocessing procedure (for example on ljspeech dataset). In case you want training pwgan with tensorflow, let take a look below instruction, it's not fully testing yet, we tried to train around 150k steps and everything is fine. 

## Training Parallel WaveGAN from scratch with LJSpeech dataset.
This example code show you how to train Parallel WaveGAN from scratch with Tensorflow 2 based on custom training loop and tf.function. The data used for this example is LJSpeech, you can download the dataset at  [link](https://keithito.com/LJ-Speech-Dataset/).

### Step 1: Create Tensorflow based Dataloader (tf.dataset)
Please see detail at [examples/melgan/](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/melgan#step-1-create-tensorflow-based-dataloader-tfdataset)

### Step 2: Training from scratch
After you re-define your dataloader, pls modify an input arguments, train_dataset and valid_dataset from [`train_parallel_wavegan.py`](https://github.com/tensorspeech/TensorFlowTTS/tree/master/examples/parallel_wavegan/train_parallel_wavegan.py). Here is an example command line to training Parallel WaveGAN from scratch:

First, you need training generator 100K steps with only stft loss:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/parallel_wavegan/train_parallel_wavegan.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./examples/parallel_wavegan/exp/train.parallel_wavegan.v1/ \
  --config ./examples/parallel_wavegan/conf/parallel_wavegan.v1.yaml \
  --use-norm 1 \
  --generator_mixed_precision 1 \
  --resume ""
```

Then resume and start training generator + discriminator:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/parallel_wavegan/parallel_wavegan.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./examples/parallel_wavegan/exp/train.parallel_wavegan.v1/ \
  --config ./examples/parallel_wavegan/conf/parallel_wavegan.v1.yaml \
  --use-norm 1 \
  --resume ./examples/parallel_wavegan/exp/train.parallel_wavegan.v1/checkpoints/ckpt-100000
```

IF you want to use MultiGPU to training you can replace `CUDA_VISIBLE_DEVICES=0` by `CUDA_VISIBLE_DEVICES=0,1,2,3` for example. You also need to tune the `batch_size` for each GPU (in config file) by yourself to maximize the performance. Note that MultiGPU now support for Training but not yet support for Decode. 

In case you want to resume the training progress, please following below example command line:

```bash
--resume ./examples/parallel_wavegan/exp/train.parallel_wavegan.v1/checkpoints/ckpt-100000
```

### Step 3: Decode audio from folder mel-spectrogram
To running inference on folder mel-spectrogram (eg valid folder), run below command line:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/parallel_wavegan/decode_parallel_wavegan.py \
  --rootdir ./dump/valid/ \
  --outdir ./prediction/parallel_wavegan.v1/ \
  --checkpoint ./examples/parallel_wavegan/exp/train.parallel_wavegan.v1/checkpoints/generator-400000.h5 \
  --config ./examples/parallel_wavegan/conf/parallel_wavegan.v1.yaml \
  --batch-size 32 \
  --use-norm 1
```

## Finetune Parallel WaveGAN with ljspeech pretrained on other languages
Just load pretrained model and training from scratch with other languages. **DO NOT FORGET** re-preprocessing on your dataset if needed. A hop_size should be 256 if you want to use our pretrained.


## Reference

1. https://github.com/kan-bayashi/ParallelWaveGAN
2. [Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram](https://arxiv.org/abs/1910.11480)