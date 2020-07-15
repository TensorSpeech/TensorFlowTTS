# Tacotron 2
Based on the script [`train_tacotron2.py`](https://github.com/dathudeptrai/TensorflowTTS/blob/master/examples/tacotron2/train_tacotron2.py).

## Training Tacotron 2 from scratch with LJSpeech dataset

This example code shows you how to train Tacotron 2 from scratch with TensorFlow 2 based on a custom training loop and [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function). The data used for this example is the LJSpeech dataset, which you can download at [[1]](https://keithito.com/LJ-Speech-Dataset).

### Step 1: Create TensorFlow based data loader (tf.dataset)

The data loader is based on the `AbstractDataset` class (see [`abstract_dataset.py`](https://github.com/dathudeptrai/TensorflowTTS/blob/master/tensorflow_tts/datasets/abstract_dataset.py)). You can choose to build a new one, or modify the given one with your dataset requirements. If using the provided preprocessing tools, the changes should be minimal, but take into account that character and mel files are found using suffix names (see [`tacotron_dataset.py`](https://github.com/dathudeptrai/TensorflowTTS/blob/master/examples/tacotron2/tacotron_dataset.py)). The generator function for the dataset should return `[character_ids, char_length, mel, mel_length]` and guided attention is recommended (see [[2]](https://arxiv.org/abs/1710.08969)).

### Step 2: Training from scratch

After adapting the data loader, you need to check and modify the input arguments for the training and validation dataset parameters in [`train_tacotron2.py`](https://github.com/dathudeptrai/TensorflowTTS/blob/master/examples/tacotron2/train_tacotron2.py). Here is an example command line to train Tacotron 2 from scratch:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/train_tacotron2.py \
  --train_dir ./dump/train \
  --valid_dir ./dump/valid \
  --outdir ./examples/tacotron2/exp/train.tacotron2.v1 \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --use_norm \
  --mixed_precision \
  --resume ""
```

If you want to use MultiGPU to training you can replace `CUDA_VISIBLE_DEVICES=0` by `CUDA_VISIBLE_DEVICES=0,1,2,3` for example. You also need to tune the `batch_size` for each GPU (in config file) by yourself to maximize the performance. Note that MultiGPU has support for the training but not yet for decoding.

In case you want to resume the training process, please follow the example below:

```bash
--resume ./examples/tacotron2/exp/train.tacotron2.v1/checkpoints/ckpt-100000
```

### Step 3: Decode mel-spectrogram from folder ids
To running inference on folder ids (charactor), run below command line:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/decode_tacotron2.py \
  --rootdir ./dump/valid/ \
  --outdir ./prediction/tacotron2-120k/ \
  --checkpoint ./examples/tacotron2/exp/train.tacotron2.v1/checkpoints/model-120000.h5 \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --batch_size 32
```

Or to decode sentences in a file:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/decode_tacotron2.py \
  --rootdir ./dump/sentences.txt \
  --outdir ./prediction/tacotron2-120k \
  --checkpoint ./examples/tacotron2/exp/train.tracotron2.v1/checkpoints/model-120000.h5 \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --batch_size 32
```

### Step 4: Extract duration from alignments for FastSpeech

You may need to extract durations for student models like FastSpeech. Here we use teacher forcing with window masking trick to extract durations from alignment maps:

Extract for validation dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/extract_duration.py \
  --rootdir ./dump/valid \
  --outdir ./dump/valid/durations \
  --checkpoint ./examples/tacotron2/exp/train.tracotron2.v1/checkpoints/model-120000.h5 \
  --use_norm \
  --stats_path ./dump/stats.npy \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --batch_size 32 \
  --win_front 3 \
  --win_back 3
```

Extract for training dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/extract_duration.py \
  --rootdir ./dump/train \
  --outdir ./dump/train/durations \
  --checkpoint ./examples/tacotron2/exp/train.tracotron2.v1/checkpoints/model-120000.h5 \
  --use_norm \
  --stats_path ./dump/stats.npy \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --batch_size 32 \
  --win_front 3 \
  --win_back 3
```

You also can download my extracted durations at 40k steps at [link](https://drive.google.com/drive/u/1/folders/1kaPXRdLg9gZrll9KtvH3-feOBMM8sn3_?usp=drive_open).

## Finetune Tacotron 2 with ljspeech pretrained on other languages

Here is an example on how to use pretrained LJSpeech models to train on other languages. This does not guarantee a better model or faster convergence in all cases, but it will improve if there is a correlation between target language and pretrained language. The only thing you need to do before finetuning on other languages is redefine embedding layers. You can do it with the following code:

```python
tacotron_config = Tacotron2Config(**config["tacotron2_params"])
tacotron_config.vocab_size = NEW_VOCAB_SIZE
tacotron2 = TFTacotron2(config=tacotron_config, training=True, name="tacotron2")
tacotron2._build()
tacotron2.summary()
tacotron2.load_weights(
    "./examples/tacotron2/exp/train.tacotron2.v1/checkpoints/model-120000.h5",
    by_name=True,
    skip_mismatch=True
)
...  # training as normal
```

## Results
Here is a result of Tacotron 2 based on this config [`tacotron2.v1.yaml`](https://github.com/dathudeptrai/TensorflowTTS/blob/tacotron-2-example/examples/tacotron-2/conf/tacotron2.v1.yaml) but with `reduction_factor=7`, we will update learning curves for `reduction_factor=1`.

### Alignments progress

<img src="fig/alignment.gif" height="300">

### Learning curves

<img src="fig/tensorboard.png" height="500">

## Some important notes

* This implementation uses guided attention by default to help the model learn diagonal alignment faster. After 15-20k, you can disable alignment loss.
* ReLU activation function works better than Mish and others.
* Supports window masking for inference, which solves problems with very long sentences.
* The model converges at around 100k steps.
* Scheduled teacher forcing is supported but training with teacher forcing gives the best performance based on my experiments. You need to be aware of the importance of applying high dropout for pre-net (both for training and inference). This will reduce the effect of previous mel prediction, so in an inference stage, a noise of prev mel prediction will not affect too much to a current decoder.
* If the amplitude levels of synthesis audio is lower compared to the original speech, you may need to multiply mel prediction with a global gain constant (e.g. 1.2).
* Using `input_signature` for Tacotron 2 makes training slower, don't know why, so only use `experimental_relax_shapes=True`.
* The implementation supports both variable and fixed shape batches, meaning that batches will be padded to the largest element in the batch or the largest in the dataset, respectively. Use `use_fixed_shapes: false` in the config file to change the behavior.

## Pretrained Models and Audio samples

| Model | Conf  | Lang  | Fs [Hz] | Mel range [Hz] | FFT / Hop / Win [pt] | # iters | Reduction factor|
| :---- | :---: | :---: | :-----: | :------------: | :------------------: | :-----: | :-------------: |
| [tacotron2.v1](https://drive.google.com/open?id=1kaPXRdLg9gZrll9KtvH3-feOBMM8sn3_) | [link](https://github.com/dathudeptrai/TensorflowTTS/tree/master/examples/tacotron2/conf/tacotron2.v1.yaml) | EN | 22.05k | 80-7600 | 1024 / 256 / None | 65k | 1 |

## References

[1] Keith Ito. "The LJ Speech Dataset". (2017). https://keithito.com/LJ-Speech-Dataset.

[2] H. Tachibana, K. Uenoyama, and S. Aihara. "Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention". (2017). arXiv: [1710.08969 [cs.SD]](https://arxiv.org/abs/1710.08969).

[3] Jonathan Shen et al. "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions". (2017). arXiv: [1712.05884 [cs.CL]](https://arxiv.org/abs/1712.05884).

[4] Alex Graves. "Generating Sequences With Recurrent Neural Networks". (2013). arXiv: [1308.0850 [cs.NE]](https://arxiv.org/abs/1308.0850).

[5] Rayhane Mama. "Tacotron-2". URL: https://github.com/Rayhane-mamah/Tacotron-2.

[6] Mozilla (Eren Gölge). "TTS". URL: https://github.com/mozilla/TTS.

[7] Tomoki Hayashi et al. "ESPnet-TTS: Unified, Reproducible, and Integratable Open Source End-to-End Text-to-Speech Toolkit". (2019). arXiv:[1910.10909 [cs.CL]](https://arxiv.org/abs/1910.10909v2). URL: https://github.com/espnet/espnet.

[8] "TensorFlow Addons". URL: https://github.com/tensorflow/addons.