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

import argparse
import logging
import os
import sys
sys.path.append(".")

import numpy as np
import soundfile as sf
import tensorflow as tf
import yaml

import tensorflow_tts

from tqdm import tqdm

from tensorflow_tts.trainers import GanBasedTrainer

from examples.melgan.audio_mel_dataset import AudioMelDataset

from tensorflow_tts.models import TFMelGANGenerator
from tensorflow_tts.models import TFMelGANMultiScaleDiscriminator

from tensorflow_tts.losses import TFMelSpectrogram

import tensorflow_tts.configs.melgan as MELGAN_CONFIG


class MelganTrainer(GanBasedTrainer):
    """Melgan Trainer class based on GanBasedTrainer."""

    def __init__(
        self,
        config,
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
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.mae_loss = tf.keras.losses.MeanAbsoluteError()
        self.mels_loss = TFMelSpectrogram()

    def init_train_eval_metrics(self, list_metrics_name):
        """Init train and eval metrics to save it to tensorboard."""
        self.train_metrics = {}
        self.eval_metrics = {}
        for name in list_metrics_name:
            self.train_metrics.update(
                {name: tf.keras.metrics.Mean(name="train_" + name, dtype=tf.float32)}
            )
            self.eval_metrics.update(
                {name: tf.keras.metrics.Mean(name="eval_" + name, dtype=tf.float32)}
            )

    def reset_states_train(self):
        """Reset train metrics after save it to tensorboard."""
        for metric in self.train_metrics.keys():
            self.train_metrics[metric].reset_states()

    def reset_states_eval(self):
        """Reset eval metrics after save it to tensorboard."""
        for metric in self.eval_metrics.keys():
            self.eval_metrics[metric].reset_states()

    def _train_step(self, batch):
        """Train model one step."""
        y, mels = batch
        y, y_hat = self._one_step_generator(y, mels)
        self._one_step_discriminator(y, y_hat)

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    @tf.function(experimental_relax_shapes=True)
    def _one_step_generator(self, y, mels):
        """One step generator training."""
        with tf.GradientTape() as g_tape:
            y_hat = self.generator(mels)  # [B, T, 1]
            p_hat = self.discriminator(y_hat)
            adv_loss = 0.0
            for i in range(len(p_hat)):
                adv_loss += self.mse_loss(
                    p_hat[i][-1], tf.ones_like(p_hat[i][-1], dtype=tf.float32)
                )
            adv_loss /= i + 1

            p = self.discriminator(tf.expand_dims(y, 2))
            # define feature-matching loss
            fm_loss = 0.0
            for i in range(len(p_hat)):
                for j in range(len(p_hat[i]) - 1):
                    fm_loss += self.mae_loss(p_hat[i][j], p[i][j])
            fm_loss /= (i + 1) * (j + 1)
            adv_loss += self.config["lambda_feat_match"] * fm_loss
            gen_loss = adv_loss

            if self.is_generator_mixed_precision:
                scaled_gen_loss = self.gen_optimizer.get_scaled_loss(gen_loss)

        if self.is_generator_mixed_precision:
            scaled_gradients = g_tape.gradient(
                scaled_gen_loss, self.generator.trainable_variables
            )
            gradients = self.gen_optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = g_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        # accumulate loss into metrics
        self.train_metrics["adversarial_loss"].update_state(adv_loss)
        self.train_metrics["fm_loss"].update_state(fm_loss)
        self.train_metrics["gen_loss"].update_state(gen_loss)
        self.train_metrics["mels_spectrogram_loss"].update_state(
            self.mels_loss(y, tf.squeeze(y_hat, -1))
        )

        # recompute y_hat after 1 step generator for discriminator training.
        y_hat = self.generator(mels)
        return y, y_hat

    @tf.function(experimental_relax_shapes=True)
    def _one_step_discriminator(self, y, y_hat):
        """One step discriminator training."""
        with tf.GradientTape() as d_tape:
            y = tf.expand_dims(y, 2)
            p = self.discriminator(y)
            p_hat = self.discriminator(y_hat)
            real_loss = 0.0
            fake_loss = 0.0
            for i in range(len(p)):
                real_loss += self.mse_loss(
                    p[i][-1], tf.ones_like(p[i][-1], dtype=tf.float32)
                )
                fake_loss += self.mse_loss(
                    p_hat[i][-1], tf.zeros_like(p_hat[i][-1], dtype=tf.float32)
                )
            real_loss /= i + 1
            fake_loss /= i + 1
            dis_loss = real_loss + fake_loss

            if self.is_discriminator_mixed_precision:
                scaled_dis_loss = self.dis_optimizer.get_scaled_loss(dis_loss)

        if self.is_discriminator_mixed_precision:
            scaled_gradients = d_tape.gradient(
                scaled_dis_loss, self.discriminator.trainable_variables
            )
            gradients = self.dis_optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = d_tape.gradient(
                dis_loss, self.discriminator.trainable_variables
            )
        self.dis_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )

        # accumulate loss into metrics
        self.train_metrics["real_loss"].update_state(real_loss)
        self.train_metrics["fake_loss"].update_state(fake_loss)
        self.train_metrics["dis_loss"].update_state(dis_loss)

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.eval_data_loader, desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(batch)

            if eval_steps_per_epoch <= self.config["num_save_intermediate_results"]:
                # save intermedia
                self.generate_and_save_intermediate_result(batch)

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.eval_metrics.keys():
            logging.info(
                f"(Steps: {self.steps}) eval_{key} = {self.eval_metrics[key].result():.4f}."
            )

        # record
        self._write_to_tensorboard(self.eval_metrics, stage="eval")

        # reset
        self.reset_states_eval()

    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        """Evaluate model one step."""
        y, mels = batch  # [B, T], [B, T, 80]

        # Generator
        y_hat = self.generator(mels)
        p_hat = self.discriminator(y_hat)
        adv_loss = 0.0
        for i in range(len(p_hat)):
            adv_loss += self.mse_loss(
                p_hat[i][-1], tf.ones_like(p_hat[i][-1], dtype=tf.float32)
            )
        adv_loss /= i + 1

        p = self.discriminator(tf.expand_dims(y, 2))
        fm_loss = 0.0
        for i in range(len(p_hat)):
            for j in range(len(p_hat[i]) - 1):
                fm_loss += self.mae_loss(p_hat[i][j], p[i][j])

        fm_loss /= (i + 1) * (j + 1)
        gen_loss = adv_loss + self.config["lambda_feat_match"] * fm_loss

        # discriminator
        p_hat = self.discriminator(y_hat)
        real_loss = 0.0
        fake_loss = 0.0
        for i in range(len(p)):
            real_loss += self.mse_loss(p[i][-1], tf.ones_like(p[i][-1], tf.float32))
            fake_loss += self.mse_loss(
                p_hat[i][-1], tf.zeros_like(p_hat[i][-1], tf.float32)
            )
        real_loss /= i + 1
        fake_loss /= i + 1
        dis_loss = real_loss + fake_loss

        # add to total eval loss
        self.eval_metrics["adversarial_loss"].update_state(adv_loss)
        self.eval_metrics["fm_loss"].update_state(fm_loss)
        self.eval_metrics["gen_loss"].update_state(gen_loss)
        self.eval_metrics["real_loss"].update_state(real_loss)
        self.eval_metrics["fake_loss"].update_state(fake_loss)
        self.eval_metrics["dis_loss"].update_state(dis_loss)
        self.eval_metrics["mels_spectrogram_loss"].update_state(
            self.mels_loss(y, tf.squeeze(y_hat, -1))
        )

    def _check_log_interval(self):
        """Log to tensorboard."""
        if self.steps % self.config["log_interval_steps"] == 0:
            for metric_name in self.list_metrics_name:
                logging.info(
                    f"(Step: {self.steps}) train_{metric_name} = {self.train_metrics[metric_name].result():.4f}."
                )
            self._write_to_tensorboard(self.train_metrics, stage="train")

            # reset
            self.reset_states_train()

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[tf.TensorSpec([None, None, 80])],
    )
    def predict(self, mels):
        """Predict."""
        return self.generator(mels)

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # generate
        y_batch, x_batch = batch
        y_batch_ = self.predict(x_batch)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 1):
            # convert to ndarray
            y, y_ = tf.reshape(y, [-1]).numpy(), tf.reshape(y_, [-1]).numpy()

            # plit figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
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

    def _check_train_finish(self):
        """Check training finished."""
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True

    def fit(self, train_data_loader, valid_data_loader, saved_path, resume=None):
        self.set_train_data_loader(train_data_loader)
        self.set_eval_data_loader(valid_data_loader)
        self.create_checkpoint_manager(saved_path=saved_path, max_to_keep=10000)
        if len(resume) > 1:
            self.load_checkpoint(resume)
            logging.info(f"Successfully resumed from {resume}.")
        self.run()


def collater(
    audio,
    mel,
    batch_max_steps=tf.constant(8192, dtype=tf.int32),
    hop_size=tf.constant(256, dtype=tf.int32),
):
    """Initialize collater (mapping function) for Tensorflow Audio-Mel Dataset.

    Args:
        batch_max_steps (int): The maximum length of input signal in batch.
        hop_size (int): Hop size of auxiliary features.

    """
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

    return audio, mel


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
    args = parser.parse_args()

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
        ] + 2 * config["generator_params"].get("aux_context_window", 0)
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
        map_fn=lambda a, b: collater(
            a, b, batch_max_steps=tf.constant(config["batch_max_steps"], dtype=tf.int32)
        ),
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"],
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
        map_fn=lambda a, b: collater(
            a,
            b,
            batch_max_steps=tf.constant(
                config["batch_max_steps_valid"], dtype=tf.int32
            ),
        ),
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"],
    )

    # define generator and discriminator
    generator = TFMelGANGenerator(
        MELGAN_CONFIG.MelGANGeneratorConfig(**config["generator_params"]),
        name="melgan_generator",
    )

    discriminator = TFMelGANMultiScaleDiscriminator(
        MELGAN_CONFIG.MelGANDiscriminatorConfig(**config["discriminator_params"]),
        name="melgan_discriminator",
    )

    # dummy input to build model.
    fake_mels = tf.random.uniform(shape=[1, 100, 80], dtype=tf.float32)
    y_hat = generator(fake_mels)
    discriminator(y_hat)

    generator.summary()
    discriminator.summary()

    # Load pretrained here
    # and fine-tune on ur target dataset.
    # generater.load_weights(pretrained_generator.h5)
    # discriminator.load_weights(pretrained_discriminator.h5)

    # define trainer
    trainer = MelganTrainer(
        steps=0,
        epochs=0,
        config=config,
        is_generator_mixed_precision=args.generator_mixed_precision,
        is_discriminator_mixed_precision=args.discriminator_mixed_precision,
    )

    trainer.compile(
        gen_model=generator,
        dis_model=discriminator,
        gen_optimizer=tf.keras.optimizers.Adam(**config["generator_optimizer_params"]),
        dis_optimizer=tf.keras.optimizers.Adam(
            **config["discriminator_optimizer_params"]
        ),
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
