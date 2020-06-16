# FastSpeech 2: Fast and High-Quality End-to-End Text to Speech
Based on the script [`train_fastspeech2.py`]().

## Training FastSpeech2 from scratch with LJSpeech dataset.
This example code show you how to train FastSpeech from scratch with Tensorflow 2 based on custom training loop and tf.function. The data used for this example is LJSpeech, you can download the dataset at  [link](https://keithito.com/LJ-Speech-Dataset/).

### Step 1: Create Tensorflow based Dataloader (tf.dataset)
First, you need define data loader based on AbstractDataset class (see [`abstract_dataset.py`](https://github.com/dathudeptrai/TensorflowTTS/tree/master/tensorflow_tts/datasets/abstract_dataset.py)). On this example, a dataloader read dataset from path. I use suffix to classify what file is a charactor, duration and mel-spectrogram (see [`fastspeech2_dataset.py`](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/fastspeech/fastspeech_dataset.py)). If you already have preprocessed version of your target dataset, you don't need to use this example dataloader, you just need refer my dataloader and modify **generator function** to adapt with your case. Normally, a generator function should return [charactor_ids, duration, f0, energy, mel]. Pls see tacotron2-example to know how to extract durations [Extract Duration](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/tacotron2#step-4-extract-duration-from-alignments-for-fastspeech)

### Step 2: Training from scratch
After you redefine your dataloader, pls modify an input arguments, train_dataset and valid_dataset from [`train_fastspeech2.py`](). Here is an example command line to training fastspeech from scratch:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2/train_fastspeech2.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./examples/fastspeech2/exp/train.fastspeech2.v1/ \
  --config ./examples/fastspeech2/conf/fastspeech2.v1.yaml \
  --use-norm 1 \
  --f0-stat ./dump/stats_f0.npy \
  --energy-stat ./dump/stats_energy.npy \
  --mixed_precision 0 \
  --resume ""
```

### Step 3: Decode mel-spectrogram from folder ids
Coming soon ...

## Finetune FastSpeech with ljspeech pretrained on other languages
Coming soon ...
## Results
Coming soon ...

### Learning curves
Coming soon ...

## Some important notes
	
* Duration/Energy/F0 predictor shared model structure but not model parameters.
* I apply log-scale bins for both Energy and F0.
* With groundtruth F0 and Energy, the mel-reconstruction loss decrease significantly.

## Pretrained Models and Audio samples
Coming soon ...

## Reference

1. [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)