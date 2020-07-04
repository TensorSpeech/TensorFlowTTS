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
"""Train FastSpeech2."""

from tensorflow_tts.optimizers import AdamWeightDecay
from tensorflow_tts.optimizers import WarmUp
from tensorflow_tts.models import TFFastSpeech2
from tensorflow_tts.configs import FastSpeech2Config
from examples.fastspeech2.fastspeech2_dataset import CharactorDurationF0EnergyMelDataset
from examples.fastspeech.train_fastspeech import FastSpeechTrainer
from tqdm import tqdm
import tensorflow_tts
import yaml
import tensorflow as tf
import numpy as np
import argparse
import logging
import os
import sys
sys.path.append(".")


class FastSpeech2Trainer(FastSpeechTrainer):
    """FastSpeech2 Trainer class based on FastSpeechTrainer."""

    def __init__(
        self, config, steps=0, epochs=0, is_mixed_precision=False,
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_mixed_precision (bool): Use mixed precision or not.

        """
        super(FastSpeech2Trainer, self).__init__(
            steps=steps,
            epochs=epochs,
            config=config,
            is_mixed_precision=is_mixed_precision,
        )
        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "duration_loss",
            "f0_loss",
            "energy_loss",
            "mel_loss_before",
            "mel_loss_after",
        ]
        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

        self.config = config

    def compile(self, model, optimizer):
        super().compile(model, optimizer)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()

    def _train_step(self, batch):
        """Train model one step."""
        charactor, duration, f0, energy, mel = batch
        self._one_step_fastspeech2(charactor, duration, f0, energy, mel)

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec([None, None], dtype=tf.int32),
            tf.TensorSpec([None, None], dtype=tf.int32),
            tf.TensorSpec([None, None], dtype=tf.float32),
            tf.TensorSpec([None, None], dtype=tf.float32),
            tf.TensorSpec([None, None, 80], dtype=tf.float32),
        ],
    )
    def _one_step_fastspeech2(self, charactor, duration, f0, energy, mel):
        with tf.GradientTape() as tape:
            (
                mel_before,
                mel_after,
                duration_outputs,
                f0_outputs,
                energy_outputs,
            ) = self.model(
                charactor,
                attention_mask=tf.math.not_equal(charactor, 0),
                speaker_ids=tf.zeros(shape=[tf.shape(mel)[0]]),
                duration_gts=duration,
                f0_gts=f0,
                energy_gts=energy,
                training=True,
            )
            log_duration = tf.math.log(tf.cast(tf.math.add(duration, 1), tf.float32))
            duration_loss = self.mse(log_duration, duration_outputs)
            f0_loss = self.mse(f0, f0_outputs)
            energy_loss = self.mse(energy, energy_outputs)
            mel_loss_before = self.mae(mel, mel_before)
            mel_loss_after = self.mae(mel, mel_after)
            loss = (
                duration_loss + f0_loss + energy_loss + mel_loss_before + mel_loss_after
            )

            if self.is_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(
                scaled_loss, self.model.trainable_variables
            )
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables), 5.0
        )

        # accumulate loss into metrics
        self.train_metrics["duration_loss"].update_state(duration_loss)
        self.train_metrics["f0_loss"].update_state(f0_loss)
        self.train_metrics["energy_loss"].update_state(energy_loss)
        self.train_metrics["mel_loss_before"].update_state(mel_loss_before)
        self.train_metrics["mel_loss_after"].update_state(mel_loss_after)

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.eval_data_loader, desc="[eval]"), 1
        ):
            # eval one step
            charactor, duration, f0, energy, mel = batch
            self._eval_step(charactor, duration, f0, energy, mel)

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

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec([None, None], dtype=tf.int32),
            tf.TensorSpec([None, None], dtype=tf.int32),
            tf.TensorSpec([None, None], dtype=tf.float32),
            tf.TensorSpec([None, None], dtype=tf.float32),
            tf.TensorSpec([None, None, 80], dtype=tf.float32),
        ],
    )
    def _eval_step(self, charactor, duration, f0, energy, mel):
        """Evaluate model one step."""
        (
            mel_before,
            mel_after,
            duration_outputs,
            f0_outputs,
            energy_outputs,
        ) = self.model(
            charactor,
            attention_mask=tf.math.not_equal(charactor, 0),
            speaker_ids=tf.zeros(shape=[tf.shape(mel)[0]]),
            duration_gts=duration,
            f0_gts=f0,
            energy_gts=energy,
            training=False,
        )
        log_duration = tf.math.log(tf.cast(tf.math.add(duration, 1), tf.float32))
        duration_loss = self.mse(log_duration, duration_outputs)
        f0_loss = self.mse(f0, f0_outputs)
        energy_loss = self.mse(energy, energy_outputs)
        mel_loss_before = self.mae(mel, mel_before)
        mel_loss_after = self.mae(mel, mel_after)

        # accumulate loss into metrics
        self.eval_metrics["duration_loss"].update_state(duration_loss)
        self.eval_metrics["f0_loss"].update_state(f0_loss)
        self.eval_metrics["energy_loss"].update_state(energy_loss)
        self.eval_metrics["mel_loss_before"].update_state(mel_loss_before)
        self.eval_metrics["mel_loss_after"].update_state(mel_loss_after)

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
        input_signature=[
            tf.TensorSpec([None, None], dtype=tf.int32),
            tf.TensorSpec([None, None], dtype=tf.int32),
            tf.TensorSpec([None, None], dtype=tf.float32),
            tf.TensorSpec([None, None], dtype=tf.float32),
            tf.TensorSpec([None, None, 80], dtype=tf.float32),
        ],
    )
    def predict(self, charactor, duration, f0, energy, mel):
        """Predict."""
        mel_before, mel_after, _, _, _ = self.model(
            charactor,
            attention_mask=tf.math.not_equal(charactor, 0),
            speaker_ids=tf.zeros(shape=[tf.shape(mel)[0]]),
            duration_gts=duration,
            f0_gts=f0,
            energy_gts=energy,
            training=False,
        )
        return mel_before, mel_after

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # unpack input.
        charactor, duration, f0, energy, mel = batch

        # predict with tf.function.
        masked_mel_before, masked_mel_after = self.predict(
            charactor, duration, f0, energy, mel
        )

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (mel_gt, mel_pred_before, mel_pred_after) in enumerate(
            zip(mel, masked_mel_before, masked_mel_after), 1
        ):
            mel_gt = tf.reshape(mel_gt, (-1, 80)).numpy()  # [length, 80]
            mel_pred_before = tf.reshape(
                mel_pred_before, (-1, 80)
            ).numpy()  # [length, 80]
            mel_pred_after = tf.reshape(
                mel_pred_after, (-1, 80)
            ).numpy()  # [length, 80]

            # plit figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            im = ax1.imshow(np.rot90(mel_gt), aspect="auto", interpolation="none")
            ax1.set_title("Target Mel-Spectrogram")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
            ax2.set_title("Predicted Mel-before-Spectrogram")
            im = ax2.imshow(
                np.rot90(mel_pred_before), aspect="auto", interpolation="none"
            )
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)
            ax3.set_title("Predicted Mel-after-Spectrogram")
            im = ax3.imshow(
                np.rot90(mel_pred_after), aspect="auto", interpolation="none"
            )
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax3)
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train FastSpeech (See detail in tensorflow_tts/bin/train-fastspeech.py)"
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
        "--use-norm", default=1, type=int, help="usr norm-mels for train or raw."
    )
    parser.add_argument(
        "--f0-stat",
        default="./dump/stats_f0.npy",
        type=str,
        required=True,
        help="f0-stat path.",
    )
    parser.add_argument(
        "--energy-stat",
        default="./dump/stats_energy.npy",
        type=str,
        required=True,
        help="energy-stat path.",
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
        "--mixed_precision",
        default=0,
        type=int,
        help="using mixed precision for generator or not.",
    )
    args = parser.parse_args()

    # set mixed precision config
    if args.mixed_precision == 1:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    args.mixed_precision = bool(args.mixed_precision)
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
        raise ValueError("Please specify --valid-dir")

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
        mel_length_threshold = config["mel_length_threshold"]
    else:
        mel_length_threshold = None

    if config["format"] == "npy":
        charactor_query = "*-ids.npy"
        mel_query = "*-raw-feats.npy" if args.use_norm is False else "*-norm-feats.npy"
        duration_query = "*-durations.npy"
        f0_query = "*-raw-f0.npy"
        energy_query = "*-raw-energy.npy"
    else:
        raise ValueError("Only npy are supported.")

    # define train/valid dataset
    train_dataset = CharactorDurationF0EnergyMelDataset(
        root_dir=args.train_dir,
        charactor_query=charactor_query,
        mel_query=mel_query,
        duration_query=duration_query,
        f0_query=f0_query,
        energy_query=energy_query,
        f0_stat=args.f0_stat,
        energy_stat=args.energy_stat,
        mel_length_threshold=mel_length_threshold,
        return_utt_id=False,
    ).create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"],
    )

    valid_dataset = CharactorDurationF0EnergyMelDataset(
        root_dir=args.dev_dir,
        charactor_query=charactor_query,
        mel_query=mel_query,
        duration_query=duration_query,
        f0_query=f0_query,
        energy_query=energy_query,
        f0_stat=args.f0_stat,
        energy_stat=args.energy_stat,
        mel_length_threshold=mel_length_threshold,
        return_utt_id=False,
    ).create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"],
    )

    fastspeech = TFFastSpeech2(config=FastSpeech2Config(**config["fastspeech_params"]))
    fastspeech._build()
    fastspeech.summary()

    # define trainer
    trainer = FastSpeech2Trainer(
        config=config, steps=0, epochs=0, is_mixed_precision=args.mixed_precision
    )

    # AdamW for fastspeech
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
        decay_steps=config["optimizer_params"]["decay_steps"],
        end_learning_rate=config["optimizer_params"]["end_learning_rate"],
    )

    learning_rate_fn = WarmUp(
        initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
        decay_schedule_fn=learning_rate_fn,
        warmup_steps=int(
            config["train_max_steps"] * config["optimizer_params"]["warmup_proportion"]
        ),
    )

    optimizer = AdamWeightDecay(
        learning_rate=learning_rate_fn,
        weight_decay_rate=config["optimizer_params"]["weight_decay"],
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
    )

    # compile trainer
    trainer.compile(model=fastspeech, optimizer=optimizer)

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
