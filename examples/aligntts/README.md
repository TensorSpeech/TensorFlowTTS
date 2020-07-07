# AlignTTS: Efficient Feed-Forward Text-to-Speech System without Explicit Alignment
Based on the script [`train_fastspeech.py`](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/fastspeech/train_fastspeech.py).

## Training FastSpeech from scratch with LJSpeech dataset.
This example code show you how to train FastSpeech from scratch with Tensorflow 2 based on custom training loop and tf.function. The data used for this example is LJSpeech, you can download the dataset at  [link](https://keithito.com/LJ-Speech-Dataset/).

### Step 1: Create Tensorflow based Dataloader (tf.dataset)
First, you need define data loader based on AbstractDataset class (see [`abstract_dataset.py`](https://github.com/dathudeptrai/TensorflowTTS/tree/master/tensorflow_tts/datasets/abstract_dataset.py)). On this example, a dataloader read dataset from path. I use suffix to classify what file is a charactor, duration and mel-spectrogram (see [`fastspeech_dataset.py`](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/fastspeech/fastspeech_dataset.py)). If you already have preprocessed version of your target dataset, you don't need to use this example dataloader, you just need refer my dataloader and modify **generator function** to adapt with your case. Normally, a generator function should return [charactor_ids, duration, mel]. Pls see tacotron2-example to know how to extract durations [Extract Duration](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/tacotron2#step-4-extract-duration-from-alignments-for-fastspeech)

### Step 2: Training from scratch
After you redefine your dataloader, pls modify an input arguments, train_dataset and valid_dataset from [`train_fastspeech.py`](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/fastspeech/train_fastspeech.py). Here is an example command line to training aligntts from scratch:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/fastspeech/train_aligntts.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./examples/fastspeech/exp/aligntts.v1/ \
  --config ./examples/aligntts/conf/aligntts.v1.yaml \
  --mixed_precision 0 \
  --resume ""
```