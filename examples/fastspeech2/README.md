# FastSpeech 2: Fast and High-Quality End-to-End Text to Speech
Based on the script [`train_fastspeech2.py`](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/fastspeech2/train_fastspeech2.py).

## Training FastSpeech2 from scratch with LJSpeech dataset.
This example code show you how to train FastSpeech from scratch with Tensorflow 2 based on custom training loop and tf.function. The data used for this example is LJSpeech, you can download the dataset at  [link](https://keithito.com/LJ-Speech-Dataset/).

### Step 1: Create Tensorflow based Dataloader (tf.dataset)
First, you need define data loader based on AbstractDataset class (see [`abstract_dataset.py`](https://github.com/dathudeptrai/TensorflowTTS/tree/master/tensorflow_tts/datasets/abstract_dataset.py)). On this example, a dataloader read dataset from path. I use suffix to classify what file is a charactor, duration and mel-spectrogram (see [`fastspeech2_dataset.py`](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/fastspeech2/fastspeech2_dataset.py)). If you already have preprocessed version of your target dataset, you don't need to use this example dataloader, you just need refer my dataloader and modify **generator function** to adapt with your case. Normally, a generator function should return [charactor_ids, duration, f0, energy, mel]. Pls see tacotron2-example to know how to extract durations [Extract Duration](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/tacotron2#step-4-extract-duration-from-alignments-for-fastspeech)

### Step 2: Training from scratch
After you redefine your dataloader, pls modify an input arguments, train_dataset and valid_dataset from [`train_fastspeech2.py`](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/fastspeech2/train_fastspeech2.py). Here is an example command line to training fastspeech2 from scratch:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2/train_fastspeech2.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./examples/fastspeech2/exp/train.fastspeech2.v1/ \
  --config ./examples/fastspeech2/conf/fastspeech2.v1.yaml \
  --use-norm 1 \
  --f0-stat ./dump/stats_f0.npy \
  --energy-stat ./dump/stats_energy.npy \
  --mixed_precision 1 \
  --resume ""
```

IF you want to use MultiGPU to training you can replace `CUDA_VISIBLE_DEVICES=0` by `CUDA_VISIBLE_DEVICES=0,1,2,3` for example. You also need to tune the `batch_size` for each GPU (in config file) by yourself to maximize the performance. Note that MultiGPU now support for Training but not yet support for Decode.

In case you want to resume the training progress, please following below example command line:

```bash
--resume ./examples/fastspeech2/exp/train.fastspeech2.v1/checkpoints/ckpt-100000
```

If you want to finetune a model, use `--pretrained` like this with your model filename
```bash
--pretrained pretrained.h5
```

You can also define `var_train_expr` in config file to let model training only on some layers in case you want to fine-tune on your dataset with the same pretrained language and processor. For example, `var_train_expr: "embeddings|encoder|decoder"` means we just training all variables that `embeddings`, `encoder`, `decoder` exist in its name.


### Step 3: Decode mel-spectrogram from folder ids

```bash
CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2/decode_fastspeech2.py \
  --rootdir ./dump/valid \
  --outdir ./predictions/fastspeech2.v1/ \
  --config ./examples/fastspeech2/conf/fastspeech2.v1.yaml \
  --checkpoint ./examples/fastspeech2/checkpoints/model-150000.h5 \
  --batch-size 8
```

## What's difference ?
	
* It's not ez for the model to learn predict f0/energy on mel level as paper did. Instead, i average f0/energy based on duration to get f0/energy on charactor level then sum it into encoder_hidden_state before pass though Length-Regulator.
* I apply mean/std normalization for both f0/energy. Note that before calculate mean and std values over all training set, i remove all outliers from f0 and energy.
* Instead using 256 bins for F0 and energy as FastSpeech2 paper, i let model learn to predict real f0/energy value then pass it though one layer Conv1D with kernel_size 9 to upsamples f0/energy scalar to vector as **[FastPitch](https://arxiv.org/abs/2006.06873)** paper suggest.
* There are other modifications to make it work, let read the code carefully to make sure you won't miss anything :D.

## Pretrained Models and Audio samples
| Model                                                                                                          | Conf                                                                                                                        | Lang  | Fs [Hz] | Mel range [Hz] | FFT / Hop / Win [pt] | # iters |
| :------                                                                                                        | :---:                                                                                                                       | :---: | :----:  | :--------:     | :---------------:    | :-----: |
| [fastspeech2.v1](https://drive.google.com/drive/folders/158vFyC2pxw9xKdxp-C5WPEtgtUiWZYE0?usp=sharing)             | [link](https://github.com/TensorSpeech/TensorFlowTTS/blob/master/examples/fastspeech2/conf/fastspeech2.v1.yaml)          | EN    | 22.05k  | 80-7600        | 1024 / 256 / None    | 150k    |
| [fastspeech2.kss.v1](https://drive.google.com/drive/folders/1DU952--jVnJ5SZDSINRs7dVVSpdB7tC_?usp=sharing)             | [link](https://github.com/TensorSpeech/TensorFlowTTS/blob/master/examples/fastspeech2/conf/fastspeech2.kss.v1.yaml)          | KO    | 22.05k  | 80-7600        | 1024 / 256 / None    | 200k    |
| [fastspeech2.kss.v2](https://drive.google.com/drive/folders/1G3-AJnEsu2rYXYgo2iGIVJfCqqfbpwMu?usp=sharing)             | [link](https://github.com/TensorSpeech/TensorFlowTTS/blob/master/examples/fastspeech2/conf/fastspeech2.kss.v2.yaml)          | KO    | 22.05k  | 80-7600        | 1024 / 256 / None    | 200k    |

## Reference

1. [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
2. [FastPitch: Parallel Text-to-speech with Pitch Prediction](https://arxiv.org/abs/2006.06873)