# Training

## Folder Structure

Let's revisit the structure of our dataset now that we have new components. We need the audio files, duration files, and metadata file to be located in the same folder. Continuing with the same [sample dataset](/recipes/lightspeech-mbmelgan/dataset/#example), we should end up with these files by the end of the duration extraction step:

```
en-bookbot/
├── durations/
│   ├── en-AU-Zak_0-durations.npy
│   ├── en-AU-Zal_1-durations.npy
│   ├── ...
│   ├── en-UK-Thalia_0-durations.npy
│   ├── ...
│   ├── en-US-Madison_0-durations.npy
│   ├── ...
├── en-AU-Zak/
│   ├── en-AU-Zak_0.lab
│   ├── en-AU-Zak_0.wav
│   ├── en-AU-Zak_1.lab
│   ├── en-AU-Zak_1.wav
│   ├── ...
├── en-UK-Thalia/
│   ├── en-UK-Thalia_0.lab
│   ├── en-UK-Thalia_0.wav
│   ├── ...
├── en-US-Madison/
│   ├── en-US-Madison_0.lab
│   ├── en-US-Madison_0.wav
│   ├── ...
└── train.txt
```

## Training LightSpeech

We start by first preprocessing and normalizing the audio files. These commands will handle tokenization, feature extraction, etc.

```sh
tensorflow-tts-preprocess --rootdir ./en-bookbot --outdir ./dump --config TensorFlowTTS/preprocess/englishipa_preprocess.yaml --dataset englishipa --verbose 2
tensorflow-tts-normalize --rootdir ./dump --outdir ./dump --config TensorFlowTTS/preprocess/englishipa_preprocess.yaml --dataset englishipa --verbose 2
```

It's also recommended to fix mis-matching duration files

```sh
python TensorFlowTTS/examples/mfa_extraction/fix_mismatch.py \
  --base_path ./dump \
  --trimmed_dur_path ./en-bookbot/trimmed-durations \
  --dur_path ./en-bookbot/durations \
  --use_norm t
```

We can then train the LightSpeech model

```sh
CUDA_VISIBLE_DEVICES=0 python TensorFlowTTS/examples/lightspeech/train_lightspeech.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./lightspeech-en-bookbot \ # (1)
  --config ./TensorFlowTTS/examples/lightspeech/conf/lightspeech_englishipa.yaml \ # (2)
  --use-norm 1 \
  --f0-stat ./dump/stats_f0.npy \
  --energy-stat ./dump/stats_energy.npy \
  --mixed_precision 1 \
  --dataset_config TensorFlowTTS/preprocess/englishipa_preprocess.yaml \
  --dataset_stats dump/stats.npy \
  --dataset_mapping dump/englishipa_mapper.json
```

1. You can set this to whatever output folder you'd like.
2. This is a pre-configured training configuration. Feel free to customize it, but be careful with setting the sample rate and hop size.

Once it's finished, you should end up with the following files:

```sh
lightspeech-en-bookbot/
├── checkpoints/ # (1)
│   ├── ckpt-10000.data-00000-of-00001
│   ├── ckpt-10000.index
│   ├── ...
│   ├── model-10000.h5
│   ├── ...
│   └── model-200000.h5 # (2)
├── config.yml
├── events.out.tfevents.1669084428.bookbot-tf-2.10561.0.v2
└── predictions/ # (3)
    ├── 100000steps/
    ├── 10000steps/
    ├── ...
```

1. This contains all of the training checkpoints.
2. The final model checkpoint (which we want).
3. This contains all mid-training intermediate predictions (mel-spectrograms).

It is missing the processor file, and the final model training checkpoint is still in the `checkpoints/` subfolder. For the former, we can simply copy the file from `dump` to the output training folder. And for the latter, we can just copy the file up a directory.

```sh
cd lightspeech-en-bookbot
cp ../dump/englishipa_mapper.json processor.json
cp checkpoints/model-200000.h5 model.h5
```

## Training Multi-band MelGAN

We can then continue with the training of our Multi-band MelGAN as our Vocoder model. First of all, you have the option to either:
1. Train to generate speech from original mel-spectrogram, or
2. Train to generate speech from LightSpeech-predicted mel-spectrogram. This is also known as training on PostNets.

Selecting option 1 would likely give you a more "universal" vocoder, one that would likely retain its performance on unseen mel-spectrograms. However, I often find its performance on small-sized datasets quite poor, and hence why I'd usually opt for the second option instead. Training on PostNets would allow the model to also learn the flaws of the LightSpeech-predicted mel-spectrograms and still aim to generate the best audio quality.

To do so, we begin by extracting the PostNets of our LightSpeech models. This means running inference on all of our texts and saving the predicted mel-spectrograms. We can do so using this modified [LightSpeech PostNet Extraction Script](https://github.com/w11wo/TensorFlowTTS/blob/master/examples/lightspeech/extractls_postnets.py). With that, we can simply run

```sh
CUDA_VISIBLE_DEVICES=0 python TensorFlowTTS/examples/lightspeech/extractls_postnets.py \
  --rootdir ./dump/train \
  --outdir ./dump/train \
  --config ./TensorFlowTTS/examples/lightspeech/conf/lightspeech_englishipa.yaml \
  --checkpoint ./lightspeech-en-bookbot/model.h5 \
  --dataset_mapping ./lightspeech-en-bookbot/processor.json

CUDA_VISIBLE_DEVICES=0 python TensorFlowTTS/examples/lightspeech/extractls_postnets.py \
  --rootdir ./dump/valid \
  --outdir ./dump/valid \
  --config ./TensorFlowTTS/examples/lightspeech/conf/lightspeech_englishipa.yaml \
  --checkpoint ./lightspeech-en-bookbot/model.h5 \
  --dataset_mapping ./lightspeech-en-bookbot/processor.json
```

That will perform inference on the training and validation subsets.

Finally, we can train the Multi-band MelGAN with the HiFi-GAN Discriminator by doing the following

```sh
CUDA_VISIBLE_DEVICES=0 python TensorFlowTTS/examples/multiband_melgan_hf/train_multiband_melgan_hf.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./mb-melgan-hifi-en-bookbot/ \
  --config ./TensorFlowTTS/examples/multiband_melgan_hf/conf/multiband_melgan_hf.en.v1.yml \
  --use-norm 1 \
  --generator_mixed_precision 1 \
  --postnets 1 \
  --resume ""

CUDA_VISIBLE_DEVICES=0 python TensorFlowTTS/examples/multiband_melgan_hf/train_multiband_melgan_hf.py \
  --train-dir ./dump/train/ \
  --dev-dir ./dump/valid/ \
  --outdir ./mb-melgan-hifi-en-bookbot/ \
  --config ./TensorFlowTTS/examples/multiband_melgan_hf/conf/multiband_melgan_hf.en.v1.yml \
  --use-norm 1 \
  --postnets 1 \
  --resume ./mb-melgan-hifi-en-bookbot/checkpoints/ckpt-200000
```

Note that this first pre-trains only the generator for 200,000 steps, and then continues the remaining steps with the usual GAN training framework.

With that, we are done!