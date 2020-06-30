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
"""Train Flow-TTS."""

import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf
import yaml

import tensorflow_tts

from tqdm import tqdm

from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from examples.tacotron2.tacotron_dataset import CharactorMelDataset

from tensorflow_tts.configs import FlowTTSConfig
from tensorflow_tts.models import TFFlowTTS
from tensorflow_tts.losses import nll

from tensorflow_tts.optimizers import WarmUp
from tensorflow_tts.optimizers import AdamWeightDecay


class FlowTTSTrainer(Seq2SeqBasedTrainer):
    """FlowTTS Trainer class based on Seq2SeqBasedTrainer."""

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
        super(FlowTTSTrainer, self).__init__(
            steps=steps,
            epochs=epochs,
            config=config,
            is_mixed_precision=is_mixed_precision,
        )
        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = ["nll_loss", "ldj_loss", "length_loss"]
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
        self.nll = nll
        self.mse = tf.keras.losses.MeanSquaredError()

    def _train_step(self, batch):
        """Train model one step."""
        charactor, char_length, mel, mel_length = batch
        self._one_step_flowtts(charactor, char_length, mel, mel_length)

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    @tf.function(experimental_relax_shapes=True)
    def _one_step_flowtts(self, charactor, char_length, mel, mel_length):
        """One step training flow-tts."""
        with tf.GradientTape() as tape:
            (
                z,
                log_det_jacobian,
                zaux,
                log_likelihood,
                mel_length_predictions,
                mask,
                attention_probs,
            ) = self.model(
                charactor,
                attention_mask=tf.math.not_equal(charactor, 0),
                speaker_ids=tf.zeros(shape=[tf.shape(charactor)[0]]),
                mel_gts=mel,
                mel_lengths=mel_length,
                training=True,
            )

            # calculate all losses.
            length_loss = self.mse(
                tf.math.log(tf.cast(mel_length, dtype=tf.float32) + 1.0),
                tf.math.log(mel_length_predictions + 1.0),
            )

            nll = tf.reduce_sum(self.nll(z, mask)) - 1.0 * tf.reduce_sum(log_likelihood)
            ldj = -1.0 * tf.reduce_sum(log_det_jacobian)

            loss = length_loss + nll + ldj

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
        self.train_metrics["nll_loss"].update_state(nll)
        self.train_metrics["ldj_loss"].update_state(ldj)
        self.train_metrics["length_loss"].update_state(length_loss)

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.eval_data_loader, desc="[eval]"), 1
        ):
            # eval one step
            charactor, char_length, mel, mel_length = batch
            self._eval_step(charactor, char_length, mel, mel_length)

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
    def _eval_step(self, charactor, char_length, mel, mel_length):
        """Evaluate model one step."""
        (
            z,
            log_det_jacobian,
            zaux,
            log_likelihood,
            mel_length_predictions,
            mask,
            attention_probs,
        ) = self.model(
            charactor,
            attention_mask=tf.math.not_equal(charactor, 0),
            speaker_ids=tf.zeros(shape=[tf.shape(charactor)[0]]),
            mel_gts=mel,
            mel_lengths=mel_length,
            training=False,
        )

        # calculate all losses.
        length_loss = self.mse(
            tf.math.log(tf.cast(mel_length, dtype=tf.float32) + 1.0),
            tf.math.log(mel_length_predictions + 1.0),
        )

        nll = tf.reduce_sum(self.nll(z, mask)) - 1.0 * tf.reduce_sum(log_likelihood)
        ldj = -1.0 * tf.reduce_sum(log_det_jacobian)

        # accumulate loss into metrics
        self.eval_metrics["nll_loss"].update_state(nll)
        self.eval_metrics["ldj_loss"].update_state(ldj)
        self.eval_metrics["length_loss"].update_state(length_loss)

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
        (
            z,
            log_det_jacobian,
            zaux,
            log_likelihood,
            mel_length_predictions,
            mask,
            attention_probs,
        ) = self.model(
            charactor,
            attention_mask=tf.math.not_equal(charactor, 0),
            speaker_ids=tf.zeros(shape=[tf.shape(charactor)[0]]),
            mel_gts=mel,
            mel_lengths=mel_length,
            training=False,
        )
        return attention_probs

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # unpack input.
        charactor, char_length, mel, mel_length = batch

        # predict with tf.function for faster.
        alignments = self.predict(charactor, char_length, mel, mel_length)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, alignment in enumerate(alignments, 1):
            # plot alignment
            figname = os.path.join(dirname, f"{idx}_alignment.png")
            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.set_title(f"Alignment head-1 @ {self.steps} steps")
            ax2.set_title(f"Alignment head-2 @ {self.steps} steps")
            im1 = ax1.imshow(
                alignment.numpy()[0], aspect="auto", origin="lower", interpolation="none"
            )
            fig.colorbar(im1, ax=ax1)
            im2 = ax2.imshow(
                alignment.numpy()[1], aspect="auto", origin="lower", interpolation="none"
            )
            fig.colorbar(im2, ax=ax2)
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
        return_guided_attention=False,
        reduction_factor=1,
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
        return_guided_attention=False,
        reduction_factor=1,
        use_fixed_shapes=False,  # don't need apply fixed shape for evaluation.
    ).create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"],
    )

    flowtts_config = FlowTTSConfig(**config["flowtts_params"])
    flowtts = TFFlowTTS(config=flowtts_config, name="flowtts")
    flowtts._build()
    flowtts.summary()

    # define trainer
    trainer = FlowTTSTrainer(
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
    trainer.compile(model=flowtts, optimizer=optimizer)

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
