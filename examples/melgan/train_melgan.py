# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train MelGAN."""

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

import sys

sys.path.append(".")

import argparse
import logging
import os

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

import tensorflow_tts
import tensorflow_tts.configs.melgan as MELGAN_CONFIG
from examples.melgan.audio_mel_dataset import AudioMelDataset
from tensorflow_tts.losses import TFMelSpectrogram
from tensorflow_tts.models import TFMelGANGenerator, TFMelGANMultiScaleDiscriminator
from tensorflow_tts.trainers import GanBasedTrainer
from tensorflow_tts.utils import calculate_2d_loss, calculate_3d_loss, return_strategy


class MelganTrainer(GanBasedTrainer):
    """Melgan Trainer class based on GanBasedTrainer."""

    def __init__(
        self,
        config,
        strategy,
        steps=0,
        epochs=0,
        is_generator_mixed_precision=False,
        is_discriminator_mixed_precision=False,
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_generator_mixed_precision (bool): Use mixed precision for generator or not.
            is_discriminator_mixed_precision (bool): Use mixed precision for discriminator or not.


        """
        super(MelganTrainer, self).__init__(
            steps,
            epochs,
            config,
            strategy,
            is_generator_mixed_precision,
            is_discriminator_mixed_precision,
        )
        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "adversarial_loss",
            "fm_loss",
            "gen_loss",
            "real_loss",
            "fake_loss",
            "dis_loss",
            "mels_spectrogram_loss",
        ]
        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

        self.config = config

    def compile(self, gen_model, dis_model, gen_optimizer, dis_optimizer):
        super().compile(gen_model, dis_model, gen_optimizer, dis_optimizer)
        # define loss
        self.mse_loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mae_loss = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mels_loss = TFMelSpectrogram()

    def compute_per_example_generator_losses(self, batch, outputs):
        """Compute per example generator losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model
        
        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        audios = batch["audios"]
        y_hat = outputs

        p_hat = self._discriminator(y_hat)
        p = self._discriminator(tf.expand_dims(audios, 2))
        adv_loss = 0.0
        for i in range(len(p_hat)):
            adv_loss += calculate_3d_loss(
                tf.ones_like(p_hat[i][-1]), p_hat[i][-1], loss_fn=self.mse_loss
            )
        adv_loss /= i + 1

        # define feature-matching loss
        fm_loss = 0.0
        for i in range(len(p_hat)):
            for j in range(len(p_hat[i]) - 1):
                fm_loss += calculate_3d_loss(
                    p[i][j], p_hat[i][j], loss_fn=self.mae_loss
                )
        fm_loss /= (i + 1) * (j + 1)
        adv_loss += self.config["lambda_feat_match"] * fm_loss

        per_example_losses = adv_loss

        dict_metrics_losses = {
            "adversarial_loss": adv_loss,
            "fm_loss": fm_loss,
            "gen_loss": adv_loss,
            "mels_spectrogram_loss": calculate_2d_loss(
                audios, tf.squeeze(y_hat, -1), loss_fn=self.mels_loss
            ),
        }

        return per_example_losses, dict_metrics_losses

    def compute_per_example_discriminator_losses(self, batch, gen_outputs):
        audios = batch["audios"]
        y_hat = gen_outputs

        y = tf.expand_dims(audios, 2)
        p = self._discriminator(y)
        p_hat = self._discriminator(y_hat)

        real_loss = 0.0
        fake_loss = 0.0
        for i in range(len(p)):
            real_loss += calculate_3d_loss(
                tf.ones_like(p[i][-1]), p[i][-1], loss_fn=self.mse_loss
            )
            fake_loss += calculate_3d_loss(
                tf.zeros_like(p_hat[i][-1]), p_hat[i][-1], loss_fn=self.mse_loss
            )
        real_loss /= i + 1
        fake_loss /= i + 1
        dis_loss = real_loss + fake_loss

        # calculate per_example_losses and dict_metrics_losses
        per_example_losses = dis_loss

        dict_metrics_losses = {
            "real_loss": real_loss,
            "fake_loss": fake_loss,
            "dis_loss": dis_loss,
        }

        return per_example_losses, dict_metrics_losses

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # generate
        y_batch_ = self.one_step_predict(batch)
        y_batch = batch["audios"]
        utt_ids = batch["utt_ids"]

        # convert to tensor.
        # here we just take a sample at first replica.
        try:
            y_batch_ = y_batch_.values[0].numpy()
            y_batch = y_batch.values[0].numpy()
            utt_ids = utt_ids.values[0].numpy()
        except Exception:
            y_batch_ = y_batch_.numpy()
            y_batch = y_batch.numpy()
            utt_ids = utt_ids.numpy()

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 0):
            # convert to ndarray
            y, y_ = tf.reshape(y, [-1]).numpy(), tf.reshape(y_, [-1]).numpy()

            # plit figure and save it
            utt_id = utt_ids[idx]
            figname = os.path.join(dirname, f"{utt_id}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavefile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(
                figname.replace(".png", "_ref.wav"),
                y,
                self.config["sampling_rate"],
                "PCM_16",
            )
            sf.write(
                figname.replace(".png", "_gen.wav"),
                y_,
                self.config["sampling_rate"],
                "PCM_16",
            )


def collater(
    items,
    batch_max_steps=tf.constant(8192, dtype=tf.int32),
    hop_size=tf.constant(256, dtype=tf.int32),
):
    """Initialize collater (mapping function) for Tensorflow Audio-Mel Dataset.

    Args:
        batch_max_steps (int): The maximum length of input signal in batch.
        hop_size (int): Hop size of auxiliary features.

    """
    audio, mel = items["audios"], items["mels"]

    if batch_max_steps is None:
        batch_max_steps = (tf.shape(audio)[0] // hop_size) * hop_size

    batch_max_frames = batch_max_steps // hop_size
    if len(audio) < len(mel) * hop_size:
        audio = tf.pad(audio, [[0, len(mel) * hop_size - len(audio)]])

    if len(mel) > batch_max_frames:
        # randomly pickup with the batch_max_steps length of the part
        interval_start = 0
        interval_end = len(mel) - batch_max_frames
        start_frame = tf.random.uniform(
            shape=[], minval=interval_start, maxval=interval_end, dtype=tf.int32
        )
        start_step = start_frame * hop_size
        audio = audio[start_step : start_step + batch_max_steps]
        mel = mel[start_frame : start_frame + batch_max_frames, :]
    else:
        audio = tf.pad(audio, [[0, batch_max_steps - len(audio)]])
        mel = tf.pad(mel, [[0, batch_max_frames - len(mel)], [0, 0]])

    items = {
        "utt_ids": items["utt_ids"],
        "audios": audio,
        "mels": mel,
        "mel_lengths": len(mel),
        "audio_lengths": len(audio),
    }

    return items


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train MelGAN (See detail in tensorflow_tts/bin/train-melgan.py)"
    )
    parser.add_argument(
        "--train-dir",
        default=None,
        type=str,
        help="directory including training data. ",
    )
    parser.add_argument(
        "--dev-dir",
        default=None,
        type=str,
        help="directory including development data. ",
    )
    parser.add_argument(
        "--use-norm", default=1, type=int, help="use norm mels for training or raw."
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save checkpoints."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--generator_mixed_precision",
        default=0,
        type=int,
        help="using mixed precision for generator or not.",
    )
    parser.add_argument(
        "--discriminator_mixed_precision",
        default=0,
        type=int,
        help="using mixed precision for discriminator or not.",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        nargs="?",
        help="path of .h5 melgan generator to load weights from",
    )
    args = parser.parse_args()

    # return strategy
    STRATEGY = return_strategy()

    # set mixed precision config
    if args.generator_mixed_precision == 1 or args.discriminator_mixed_precision == 1:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    args.generator_mixed_precision = bool(args.generator_mixed_precision)
    args.discriminator_mixed_precision = bool(args.discriminator_mixed_precision)

    args.use_norm = bool(args.use_norm)

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # check arguments
    if args.train_dir is None:
        raise ValueError("Please specify --train-dir")
    if args.dev_dir is None:
        raise ValueError("Please specify either --valid-dir")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = tensorflow_tts.__version__
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    if config["remove_short_samples"]:
        mel_length_threshold = config["batch_max_steps"] // config[
            "hop_size"
        ] + 2 * config["melgan_generator_params"].get("aux_context_window", 0)
    else:
        mel_length_threshold = None

    if config["format"] == "npy":
        audio_query = "*-wave.npy"
        mel_query = "*-raw-feats.npy" if args.use_norm is False else "*-norm-feats.npy"
        audio_load_fn = np.load
        mel_load_fn = np.load
    else:
        raise ValueError("Only npy are supported.")

    # define train/valid dataset
    train_dataset = AudioMelDataset(
        root_dir=args.train_dir,
        audio_query=audio_query,
        mel_query=mel_query,
        audio_load_fn=audio_load_fn,
        mel_load_fn=mel_load_fn,
        mel_length_threshold=mel_length_threshold,
    ).create(
        is_shuffle=config["is_shuffle"],
        map_fn=lambda items: collater(
            items,
            batch_max_steps=tf.constant(config["batch_max_steps"], dtype=tf.int32),
            hop_size=tf.constant(config["hop_size"], dtype=tf.int32),
        ),
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"]
        * STRATEGY.num_replicas_in_sync
        * config["gradient_accumulation_steps"],
    )

    valid_dataset = AudioMelDataset(
        root_dir=args.dev_dir,
        audio_query=audio_query,
        mel_query=mel_query,
        audio_load_fn=audio_load_fn,
        mel_load_fn=mel_load_fn,
        mel_length_threshold=mel_length_threshold,
    ).create(
        is_shuffle=config["is_shuffle"],
        map_fn=lambda items: collater(
            items,
            batch_max_steps=tf.constant(
                config["batch_max_steps_valid"], dtype=tf.int32
            ),
            hop_size=tf.constant(config["hop_size"], dtype=tf.int32),
        ),
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
    )

    # define trainer
    trainer = MelganTrainer(
        steps=0,
        epochs=0,
        config=config,
        strategy=STRATEGY,
        is_generator_mixed_precision=args.generator_mixed_precision,
        is_discriminator_mixed_precision=args.discriminator_mixed_precision,
    )

    # define generator and discriminator
    with STRATEGY.scope():
        generator = TFMelGANGenerator(
            MELGAN_CONFIG.MelGANGeneratorConfig(**config["melgan_generator_params"]),
            name="melgan_generator",
        )

        discriminator = TFMelGANMultiScaleDiscriminator(
            MELGAN_CONFIG.MelGANDiscriminatorConfig(
                **config["melgan_discriminator_params"]
            ),
            name="melgan_discriminator",
        )

        # dummy input to build model.
        fake_mels = tf.random.uniform(shape=[1, 100, 80], dtype=tf.float32)
        y_hat = generator(fake_mels)
        discriminator(y_hat)

        if len(args.pretrained) > 1:
            generator.load_weights(args.pretrained)
            logging.info(
                f"Successfully loaded pretrained weight from {args.pretrained}."
            )

        generator.summary()
        discriminator.summary()

        gen_optimizer = tf.keras.optimizers.Adam(**config["generator_optimizer_params"])
        dis_optimizer = tf.keras.optimizers.Adam(
            **config["discriminator_optimizer_params"]
        )

    trainer.compile(
        gen_model=generator,
        dis_model=discriminator,
        gen_optimizer=gen_optimizer,
        dis_optimizer=dis_optimizer,
    )

    # start training
    try:
        trainer.fit(
            train_dataset,
            valid_dataset,
            saved_path=os.path.join(config["outdir"], "checkpoints/"),
            resume=args.resume,
        )
    except KeyboardInterrupt:
        trainer.save_checkpoint()
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
