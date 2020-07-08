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
"""Train Tacotron2."""

import argparse
import logging
import os
import sys
sys.path.append(".")

import numpy as np
import tensorflow as tf
import yaml

import tensorflow_tts

from tqdm import tqdm

from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from examples.tacotron2.tacotron_dataset import CharactorMelDataset

from tensorflow_tts.configs.tacotron2 import Tacotron2Config

from tensorflow_tts.models import TFTacotron2

from tensorflow_tts.optimizers import WarmUp
from tensorflow_tts.optimizers import AdamWeightDecay


class Tacotron2Trainer(Seq2SeqBasedTrainer):
    """Tacotron2 Trainer class based on Seq2SeqBasedTrainer."""

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
        super(Tacotron2Trainer, self).__init__(
            steps=steps,
            epochs=epochs,
            config=config,
            is_mixed_precision=is_mixed_precision,
        )
        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "stop_token_loss",
            "mel_loss_before",
            "mel_loss_after",
            "guided_attention_loss",
        ]
        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

        self.config = config

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

    def compile(self, model, optimizer):
        super().compile(model, optimizer)
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()

        # create scheduler for teacher forcing.
        self.teacher_forcing_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.config["start_ratio_value"],
            decay_steps=self.config["schedule_decay_steps"],
            end_learning_rate=self.config["end_ratio_value"],
            cycle=True,
            name="teacher_forcing_scheduler",
        )

    def _train_step(self, batch):
        """Train model one step."""
        charactor, char_length, mel, mel_length, guided_attention = batch
        self._one_step_tacotron2(
            charactor, char_length, mel, mel_length, guided_attention
        )

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()
        self._apply_schedule_teacher_forcing()

    def _apply_schedule_teacher_forcing(self):
        if self.steps >= self.config["start_schedule_teacher_forcing"]:
            # change _ratio on sampler.
            self.model.decoder.sampler._ratio = self.teacher_forcing_scheduler(
                self.steps - self.config["start_schedule_teacher_forcing"]
            )
            if self.steps == self.config["start_schedule_teacher_forcing"]:
                logging.info(
                    f"(Steps: {self.steps}) Starting apply schedule teacher forcing."
                )

    @tf.function(experimental_relax_shapes=True)
    def _one_step_tacotron2(
        self, charactor, char_length, mel, mel_length, guided_attention
    ):
        with tf.GradientTape() as tape:
            (
                mel_outputs,
                post_mel_outputs,
                stop_outputs,
                alignment_historys,
            ) = self.model(
                charactor,
                char_length,
                speaker_ids=tf.zeros(shape=[tf.shape(charactor)[0]]),
                mel_outputs=mel,
                mel_lengths=mel_length,
                training=True,
            )

            # calculate mel loss.
            mel_loss_before = self.mae(mel, mel_outputs)
            mel_loss_after = self.mae(mel, post_mel_outputs)

            # calculate stop grounth truth based-on mel_length.
            max_mel_length = (
                tf.reduce_max(mel_length)
                if self.config["use_fixed_shapes"] is False
                else [self.config["max_mel_length"]]
            )
            stop_gts = tf.expand_dims(
                tf.range(tf.reduce_max(max_mel_length), dtype=tf.int32), 0
            )  # [1, max_len]
            stop_gts = tf.tile(stop_gts, [tf.shape(mel_length)[0], 1])  # [B, max_len]
            stop_gts = tf.cast(
                tf.math.greater_equal(stop_gts, tf.expand_dims(mel_length, 1)),
                tf.float32,
            )

            stop_token_loss = self.binary_crossentropy(stop_gts, stop_outputs)

            # calculate guided attention loss.
            attention_masks = tf.cast(
                tf.math.not_equal(guided_attention, -1.0), tf.float32
            )
            loss_att = tf.reduce_sum(
                tf.abs(alignment_historys * guided_attention) * attention_masks
            )
            loss_att /= tf.reduce_sum(attention_masks)

            # sum all loss
            loss = stop_token_loss + mel_loss_before + mel_loss_after + loss_att

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
        self.train_metrics["stop_token_loss"].update_state(stop_token_loss)
        self.train_metrics["mel_loss_before"].update_state(mel_loss_before)
        self.train_metrics["mel_loss_after"].update_state(mel_loss_after)
        self.train_metrics["guided_attention_loss"].update_state(loss_att)

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")

        # set traing = False on decoder_cell
        self.model.decoder.cell.training = False

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.eval_data_loader, desc="[eval]"), 1
        ):
            # eval one step
            charactor, char_length, mel, mel_length, guided_attention = batch
            self._eval_step(charactor, char_length, mel, mel_length, guided_attention)

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

        # enable training = True on decoder_cell
        self.model.decoder.cell.training = True

    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, charactor, char_length, mel, mel_length, guided_attention):
        """Evaluate model one step."""
        mel_outputs, post_mel_outputs, stop_outputs, alignment_historys = self.model(
            charactor,
            char_length,
            speaker_ids=tf.zeros(shape=[tf.shape(charactor)[0]]),
            mel_outputs=mel,
            mel_lengths=mel_length,
            training=False,
        )

        # calculate mel loss.
        mel_loss_before = self.mae(mel, mel_outputs)
        mel_loss_after = self.mae(mel, post_mel_outputs)

        # calculate stop grounth truth based-on mel_length.
        stop_gts = tf.expand_dims(
            tf.range(tf.reduce_max(mel_length), dtype=tf.int32), 0
        )  # [1, max_len]
        stop_gts = tf.tile(stop_gts, [tf.shape(mel_length)[0], 1])  # [B, max_len]
        stop_gts = tf.cast(
            tf.math.greater_equal(stop_gts, tf.expand_dims(mel_length, 1)), tf.float32
        )

        stop_token_loss = self.binary_crossentropy(stop_gts, stop_outputs)

        # calculate guided attention loss.
        attention_masks = tf.cast(tf.math.not_equal(guided_attention, -1.0), tf.float32)
        loss_att = tf.reduce_sum(
            tf.abs(alignment_historys * guided_attention) * attention_masks
        )
        loss_att /= tf.reduce_sum(attention_masks)

        # accumulate loss into metrics
        self.eval_metrics["stop_token_loss"].update_state(stop_token_loss)
        self.eval_metrics["mel_loss_before"].update_state(mel_loss_before)
        self.eval_metrics["mel_loss_after"].update_state(mel_loss_after)
        self.eval_metrics["guided_attention_loss"].update_state(loss_att)

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

    @tf.function(experimental_relax_shapes=True)
    def predict(self, charactor, char_length, mel, mel_length):
        """Predict."""
        mel_outputs, post_mel_outputs, _, alignment = self.model(
            charactor,
            char_length,
            speaker_ids=tf.zeros(shape=[tf.shape(charactor)[0]]),
            mel_outputs=mel,
            mel_lengths=mel_length,
            training=False,
        )
        return mel_outputs, post_mel_outputs, alignment

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # unpack input.
        charactor, char_length, mel, mel_length, _ = batch

        # predict with tf.function for faster.
        masked_mel_before, masked_mel_after, alignments = self.predict(
            charactor, char_length, mel, mel_length
        )

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (mel_gt, mel_pred_before, mel_pred_after, alignment) in enumerate(
            zip(mel, masked_mel_before, masked_mel_after, alignments), 1
        ):
            mel_gt = tf.reshape(mel_gt, (-1, 80)).numpy()  # [length, 80]
            mel_pred_before = tf.reshape(
                mel_pred_before, (-1, 80)
            ).numpy()  # [length, 80]
            mel_pred_after = tf.reshape(
                mel_pred_after, (-1, 80)
            ).numpy()  # [length, 80]

            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            im = ax1.imshow(np.rot90(mel_gt), aspect="auto", interpolation="none")
            ax1.set_title("Target Mel-Spectrogram")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
            ax2.set_title(f"Predicted Mel-before-Spectrogram @ {self.steps} steps")
            im = ax2.imshow(
                np.rot90(mel_pred_before), aspect="auto", interpolation="none"
            )
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)
            ax3.set_title(f"Predicted Mel-after-Spectrogram @ {self.steps} steps")
            im = ax3.imshow(
                np.rot90(mel_pred_after), aspect="auto", interpolation="none"
            )
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax3)
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # plot alignment
            figname = os.path.join(dirname, f"{idx}_alignment.png")
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_title(f"Alignment @ {self.steps} steps")
            im = ax.imshow(
                alignment, aspect="auto", origin="lower", interpolation="none"
            )
            fig.colorbar(im, ax=ax)
            xlabel = "Decoder timestep"
            plt.xlabel(xlabel)
            plt.ylabel("Encoder timestep")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

    def _check_train_finish(self):
        """Check training finished."""
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True

    def fit(self, train_dataset, valid_dataset, saved_path, resume=None):
        self.set_train_data_loader(train_dataset)
        self.set_eval_data_loader(valid_dataset)
        self.create_checkpoint_manager(saved_path=saved_path, max_to_keep=10000)
        if len(resume) > 2:
            self.load_checkpoint(resume)
            logging.info(f"Successfully resumed from {resume}.")
        self.run()


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

    # get dataset
    if config["remove_short_samples"]:
        mel_length_threshold = config["mel_length_threshold"]
    else:
        mel_length_threshold = None

    if config["format"] == "npy":
        charactor_query = "*-ids.npy"
        mel_query = "*-raw-feats.npy" if args.use_norm is False else "*-norm-feats.npy"
        charactor_load_fn = np.load
        mel_load_fn = np.load
    else:
        raise ValueError("Only npy are supported.")

    train_dataset = CharactorMelDataset(
        root_dir=args.train_dir,
        charactor_query=charactor_query,
        mel_query=mel_query,
        charactor_load_fn=charactor_load_fn,
        mel_load_fn=mel_load_fn,
        mel_length_threshold=mel_length_threshold,
        return_utt_id=False,
        reduction_factor=config["tacotron2_params"]["reduction_factor"],
        use_fixed_shapes=config["use_fixed_shapes"],
    )

    # update max_mel_length and max_char_length to config
    config.update({"max_mel_length": int(train_dataset.max_mel_length)})
    config.update({"max_char_length": int(train_dataset.max_char_length)})

    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    train_dataset = train_dataset.create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"],
    )

    valid_dataset = CharactorMelDataset(
        root_dir=args.dev_dir,
        charactor_query=charactor_query,
        mel_query=mel_query,
        charactor_load_fn=charactor_load_fn,
        mel_load_fn=mel_load_fn,
        mel_length_threshold=mel_length_threshold,
        return_utt_id=False,
        reduction_factor=config["tacotron2_params"]["reduction_factor"],
        use_fixed_shapes=False,  # don't need apply fixed shape for evaluation.
    ).create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"],
    )

    tacotron_config = Tacotron2Config(**config["tacotron2_params"])
    tacotron2 = TFTacotron2(config=tacotron_config, training=True, name="tacotron2")
    tacotron2._build()
    tacotron2.summary()

    # define trainer
    trainer = Tacotron2Trainer(
        config=config, steps=0, epochs=0, is_mixed_precision=args.mixed_precision
    )

    # AdamW for tacotron2
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
    trainer.compile(model=tacotron2, optimizer=optimizer)

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
