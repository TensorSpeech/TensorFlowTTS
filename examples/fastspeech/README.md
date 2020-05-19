# FastSpeech: Fast, Robust and Controllable Text to Speech
Based on the script [`train_fastspeech.py`](https://github.com/dathudeptrai/TensorflowTTS/blob/fastspeech-example/examples/fastspeech/train_fastspeech.py).

## Training FastSpeech from scratch with LJSpeech dataset.
This example code show you how to train FastSpeech from scratch with Tensorflow 2 based on custom training loop and tf.function. The data used for this example is LJSpeech, you can download the dataset at  [link](https://keithito.com/LJ-Speech-Dataset/).

### Step 1: Create Tensorflow based Dataloader (tf.dataset)
First, you need define data loader based on AbstractDataset class (see [`abstract_dataset.py`](https://github.com/dathudeptrai/TensorflowTTS/blob/fastspeech-example/tensorflow_tts/datasets/abstract_dataset.py)). On this example, a dataloader read dataset from path. I use suffix to classify what file is a charactor, duration and mel-spectrogram (see [`fastspeech_dataset.py`](https://github.com/dathudeptrai/TensorflowTTS/blob/fastspeech-example/examples/fastspeech/fastspeech_dataset.py)). If you already have preprocessed version of your target dataset, you don't need to use this example dataloader, you just need refer my dataloader and modify **generator function** to adapt with your case. Normally, a generator function should return [charactor_ids, duration, mel].

### Step 2: Training from scratch
After you redefine your dataloader, pls modify an input arguments, train_dataset and valid_dataset from [`train_fastspeech.py`](https://github.com/dathudeptrai/TensorflowTTS/blob/fastspeech-example/examples/fastspeech/train_fastspeech.py). Here is an example command line to training fastspeech from scratch:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python train_fastspeech.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./exp/train.fastspeech.v1/ \
  --config conf/fastspeech.v1.yaml \
  --use-norm 1
  --mixed_precision 0 \
  --resume "" > log.fastspeech.v1.txt 2>&1
```

## Finetune FastSpeech with ljspeech pretrained on other languages
Here is an example show you how to use pretrained ljspeech to training with other languages. This does not guarantee a better model or faster convergence in all cases but it will improve if there is a correlation between target language and pretrained language. The only thing you need to do before finetune on other languages is re-define embedding layers. You can do it by following code:

```bash
pretrained_config = ...
fastspeech = TFFastSpeech(pretrained_config)
fastspeech._build()
fastspeech.summary()
fastspeech.load_weights(PRETRAINED_PATH)

# re-define here
pretrained_config.vocab_size = NEW_VOCAB_SIZE
new_embedding_layers = TFFastSpeechEmbeddings(pretrained_config, name='embeddings')
fastspeech.embeddings = new_embedding_layers
# re-build model
fastspeech._build()
fastspeech.summary()

... # training as normal.
```

## Results
Here is a result of tacotron2 based on this config [`fastspeech.v1.yaml`](https://github.com/dathudeptrai/TensorflowTTS/blob/tacotron-2-example/examples/tacotron-2/conf/tacotron2.v1.yaml)

### Learning curves
<img src="fig/tensorboard.png" height="500">

### Audio samples
You can hear some audio samples at [`audios`](https://github.com/dathudeptrai/TensorflowTTS/tree/fastspeech-example/examples/fastspeech/audios). This is FastSpeech v1 and vocoder is Melgan at 2M steps.

## Some important notes
	
* **DO NOT** apply any activation function on intermediate layer (TFFastSpeechIntermediate).
* There is no different between num_hidden_layers = 6 and num_hidden_layers = 4.
* I use mish rather than relu.

## Reference

1. https://github.com/xcmyz/FastSpeech
2. [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)