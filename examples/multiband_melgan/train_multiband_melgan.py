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
"""Train Multi-Band MelGAN."""

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

from examples.melgan.train_melgan import MelganTrainer
from examples.melgan.train_melgan import collater

from examples.melgan.audio_mel_dataset import AudioMelDataset

from tensorflow_tts.models import TFMelGANGenerator
from tensorflow_tts.models import TFMelGANMultiScaleDiscriminator
from tensorflow_tts.models import TFPQMF

from tensorflow_tts.losses import TFMultiResolutionSTFT

from tensorflow_tts.configs import MultiBandMelGANGeneratorConfig
from tensorflow_tts.configs import MultiBandMelGANDiscriminatorConfig


class MultiBandMelganTrainer(MelganTrainer):
    """Multi-Band MelGAN Trainer class based on MelganTrainer."""

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
        super(MultiBandMelganTrainer, self).__init__(
            config=config,
            steps=steps,
            epochs=epochs,
            is_generator_mixed_precision=is_generator_mixed_precision,
            is_discriminator_mixed_precision=is_discriminator_mixed_precision,
        )

        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "adversarial_loss",
            "subband_spectral_convergence_loss",
            "subband_log_magnitude_loss",
            "fullband_spectral_convergence_loss",
            "fullband_log_magnitude_loss",
            "gen_loss",
            "real_loss",
            "fake_loss",
            "dis_loss",
        ]

        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

    def compile(self, gen_model, dis_model, gen_optimizer, dis_optimizer, pqmf):
        super().compile(gen_model, dis_model, gen_optimizer, dis_optimizer)
        # define loss
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.mae_loss = tf.keras.losses.MeanAbsoluteError()
        self.sub_band_stft_loss = TFMultiResolutionSTFT(
            **self.config["subband_stft_loss_params"]
        )
        self.full_band_stft_loss = TFMultiResolutionSTFT(
            **self.config["stft_loss_params"]
        )

        # define pqmf module
        self.pqmf = pqmf

    def _train_step(self, batch):
        """Train model one step."""
        y, mels = batch

        y, y_hat = self._one_step_generator(y, mels)
        if self.steps >= self.config["discriminator_train_start_steps"]:
            self._one_step_discriminator(y, y_hat)

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    @tf.function(experimental_relax_shapes=True)
    def _one_step_generator(self, y, mels):
        """One step generator training."""
        with tf.GradientTape() as g_tape:
            y_mb_hat = self.generator(mels)  # [B, T // subbands, subbands]
            y_hat = self.pqmf.synthesis(y_mb_hat)  # [B, T, 1]

            # subband multi-resolution stft loss for multi-band model
            y_mb = self.pqmf.analysis(
                tf.expand_dims(y, -1)
            )  # [B, T//subbands, subbands]
            y_mb = tf.transpose(y_mb, (0, 2, 1))  # [B, subbands, T//subbands]
            y_mb = tf.reshape(y_mb, (-1, tf.shape(y_mb)[-1]))  # [B * subbands, T']

            y_mb_hat = tf.transpose(y_mb_hat, (0, 2, 1))  # [B, subbands, T//subbands]
            y_mb_hat = tf.reshape(
                y_mb_hat, (-1, tf.shape(y_mb_hat)[-1])
            )  # [B * subbands, T']

            # calculate sub/full band spectral_convergence and log mag loss.
            sub_sc_loss, sub_mag_loss = self.sub_band_stft_loss(y_mb, y_mb_hat)
            full_sc_loss, full_mag_loss = self.full_band_stft_loss(
                y, tf.squeeze(y_hat, -1)
            )

            # define generator loss
            gen_loss = 0.5 * (sub_sc_loss + sub_mag_loss) + 0.5 * (
                full_sc_loss + full_mag_loss
            )

            if self.steps >= self.config["discriminator_train_start_steps"]:
                p_hat = self.discriminator(y_hat)
                adv_loss = 0.0
                for i in range(len(p_hat)):
                    adv_loss += self.mse_loss(
                        p_hat[i][-1], tf.ones_like(p_hat[i][-1], dtype=tf.float32)
                    )
                adv_loss /= i + 1
                gen_loss += self.config["lambda_adv"] * adv_loss

                self.train_metrics["adversarial_loss"].update_state(adv_loss)

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
        self.train_metrics["gen_loss"].update_state(gen_loss)
        self.train_metrics["subband_spectral_convergence_loss"].update_state(
            sub_sc_loss
        )
        self.train_metrics["subband_log_magnitude_loss"].update_state(sub_mag_loss)
        self.train_metrics["fullband_spectral_convergence_loss"].update_state(
            full_sc_loss
        )
        self.train_metrics["fullband_log_magnitude_loss"].update_state(full_mag_loss)

        # recompute y_hat after 1 step generator for discriminator training.
        y_mb_hat = self.generator(mels)  # [B, T // subbands, subbands]
        y_hat = self.pqmf.synthesis(y_mb_hat)  # [B, T, 1]
        return y, y_hat

    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        """Evaluate model one step."""
        y, mels = batch  # [B, T], [B, T, 80]

        # Generator
        y_mb_hat = self.generator(mels)
        y_hat = self.pqmf.synthesis(y_mb_hat)  # [B, T, 1]

        # subband multi-resolution stft loss for multi-band model
        y_mb = self.pqmf.analysis(tf.expand_dims(y, -1))  # [B, T//subbands, subbands]
        y_mb = tf.transpose(y_mb, (0, 2, 1))  # [B, subbands, T//subbands]
        y_mb = tf.reshape(y_mb, (-1, tf.shape(y_mb)[-1]))  # [B * subbands, T']

        y_mb_hat = tf.transpose(y_mb_hat, (0, 2, 1))  # [B, subbands, T//subbands]
        y_mb_hat = tf.reshape(
            y_mb_hat, (-1, tf.shape(y_mb_hat)[-1])
        )  # [B * subbands, T']

        # calculate sub/full band spectral_convergence and log mag loss.
        sub_sc_loss, sub_mag_loss = self.sub_band_stft_loss(y_mb, y_mb_hat)
        full_sc_loss, full_mag_loss = self.full_band_stft_loss(y, tf.squeeze(y_hat, -1))

        # define gen_loss
        gen_loss = 0.5 * (sub_sc_loss + sub_mag_loss) + 0.5 * (
            full_sc_loss + full_mag_loss
        )

        if self.steps >= self.config["discriminator_train_start_steps"]:
            p_hat = self.discriminator(y_hat)
            adv_loss = 0.0
            for i in range(len(p_hat)):
                adv_loss += self.mse_loss(
                    p_hat[i][-1], tf.ones_like(p_hat[i][-1], dtype=tf.float32)
                )
            adv_loss /= i + 1
            gen_loss += adv_loss + self.config["lambda_adv"] * adv_loss

            # discriminator
            p = self.discriminator(tf.expand_dims(y, 2))
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
            self.eval_metrics["real_loss"].update_state(real_loss)
            self.eval_metrics["fake_loss"].update_state(fake_loss)
            self.eval_metrics["dis_loss"].update_state(dis_loss)

        self.eval_metrics["gen_loss"].update_state(gen_loss)
        self.eval_metrics["subband_spectral_convergence_loss"].update_state(sub_sc_loss)
        self.eval_metrics["subband_log_magnitude_loss"].update_state(sub_mag_loss)
        self.eval_metrics["fullband_spectral_convergence_loss"].update_state(
            full_sc_loss
        )
        self.eval_metrics["fullband_log_magnitude_loss"].update_state(full_mag_loss)

    @tf.function(experimental_relax_shapes=True)
    def predict(self, mels):
        """Predict."""
        return self.generator(mels)

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # generate
        y_batch, x_batch = batch
        y_mb_batch_ = self.predict(x_batch)  # [B, T // subbands, subbands]
        y_batch_ = self.pqmf.synthesis(y_mb_batch_)  # [B, T, 1]

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

        if self.steps == self.config["discriminator_train_start_steps"]:
            self.finish_train = True
            logging.info(
                f"Finished training only generator at {self.steps}steps, pls resume and continue training."
            )


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train MultiBand MelGAN (See detail in examples/multiband_melgan/train_multiband_melgan.py)"
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
        MultiBandMelGANGeneratorConfig(**config["generator_params"]),
        name="multi_band_melgan_generator",
    )

    discriminator = TFMelGANMultiScaleDiscriminator(
        MultiBandMelGANDiscriminatorConfig(**config["discriminator_params"]),
        name="multi_band_melgan_discriminator",
    )

    pqmf = TFPQMF(
        MultiBandMelGANGeneratorConfig(**config["generator_params"]), name="pqmf"
    )

    # dummy input to build model.
    fake_mels = tf.random.uniform(shape=[1, 100, 80], dtype=tf.float32)
    y_mb_hat = generator(fake_mels)
    y_hat = pqmf.synthesis(y_mb_hat)
    discriminator(y_hat)

    generator.summary()
    discriminator.summary()

    # define trainer
    trainer = MultiBandMelganTrainer(
        steps=0,
        epochs=0,
        config=config,
        is_generator_mixed_precision=args.generator_mixed_precision,
        is_discriminator_mixed_precision=args.discriminator_mixed_precision,
    )

    # define optimizer
    generator_lr_fn = getattr(
        tf.keras.optimizers.schedules, config["generator_optimizer_params"]["lr_fn"]
    )(**config["generator_optimizer_params"]["lr_params"])
    discriminator_lr_fn = getattr(
        tf.keras.optimizers.schedules, config["discriminator_optimizer_params"]["lr_fn"]
    )(**config["discriminator_optimizer_params"]["lr_params"])

    gen_optimizer = tf.keras.optimizers.Adam(
        learning_rate=generator_lr_fn,
        amsgrad=config["generator_optimizer_params"]["amsgrad"],
    )
    dis_optimizer = tf.keras.optimizers.Adam(
        learning_rate=discriminator_lr_fn,
        amsgrad=config["discriminator_optimizer_params"]["amsgrad"],
    )

    trainer.compile(
        gen_model=generator,
        dis_model=discriminator,
        gen_optimizer=gen_optimizer,
        dis_optimizer=dis_optimizer,
        pqmf=pqmf,
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
