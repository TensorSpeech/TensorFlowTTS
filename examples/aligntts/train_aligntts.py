# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen Quan Anh, Modified by Trinhlq (@l4zyf9x)
#  MIT License (https://opensource.org/licenses/MIT)

"""Train FastSpeech."""

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
from examples.aligntts.aligntts_dataset import LJSpeechDataset
from tensorflow_tts.losses.mdn import MixDensityLoss

from tensorflow_tts.configs import AlignTTSConfig

from tensorflow_tts.models import TFAlignTTS
from tensorflow_tts.models import Viterbi

from tensorflow_tts.optimizers import WarmUp
from tensorflow_tts.optimizers import AdamWeightDecay
from tensorflow_tts.utils import plot_utils
from tensorflow_tts.processor.ljspeech import _id_to_symbol


class AlignTTSTrainer(Seq2SeqBasedTrainer):
    """FastSpeech Trainer class based on Seq2SeqBasedTrainer."""

    def __init__(self,
                 config,
                 steps=0,
                 epochs=0,
                 is_mixed_precision=False,
                 ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_mixed_precision (bool): Use mixed precision or not.

        """
        super().__init__(steps=steps,
                         epochs=epochs,
                         config=config,
                         is_mixed_precision=is_mixed_precision)
        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "duration_loss",
            "energy_loss",
            "f0_loss",
            "mel_loss_after",
            "mel_loss_before",
            "attention_loss",
            "mdn_loss"
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
                {name: tf.keras.metrics.Mean(name='train_' + name, dtype=tf.float32)}
            )
            self.eval_metrics.update(
                {name: tf.keras.metrics.Mean(name='eval_' + name, dtype=tf.float32)}
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
        self.mse_log = tf.keras.losses.MeanSquaredLogarithmicError()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()
        # self.attn_measure = MaskAttentionLoss()
        self.mdn_loss = MixDensityLoss()
        self.viterbi = Viterbi()

    def _train_step(self, batch):
        """Train model one step."""
        # charactor, duration, mel, speaker_id = batch
        characters, mels, speaker_ids, character_lengths, mel_lengths = batch
        self._one_step_fastspeech(characters,
                                  mels,
                                  speaker_ids,
                                  character_lengths,
                                  mel_lengths)

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    @tf.function(experimental_relax_shapes=True,
                 input_signature=[tf.TensorSpec([None, None], dtype=tf.int32),
                                  tf.TensorSpec([None, None, 80], dtype=tf.float32),
                                  tf.TensorSpec([None], dtype=tf.int32),
                                  tf.TensorSpec([None], dtype=tf.int32),
                                  tf.TensorSpec([None], dtype=tf.int32),
                                  ])
    def _one_step_fastspeech(self,
                             characters,
                             mels,
                             speaker_ids,
                             character_lengths,
                             mel_lengths):
        with tf.GradientTape() as tape:
            masked_mel_before, masked_mel_after, \
                masked_duration_outputs, mu_sigma = self.model(
                    input_ids=characters,
                    speaker_ids=speaker_ids,
                    durations=None,
                    character_lengths=character_lengths,
                    mel_lengths=mel_lengths,
                    training=True)

            log_prob, mdn_loss, _ = self.mdn_loss((mels, mu_sigma, mel_lengths, character_lengths))
            loss = mdn_loss

            if self.is_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            if self.is_mixed_precision:
                scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
                gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                gradients = tape.gradient(loss, self.model.trainable_variables)
                gradients = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in gradients]

            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables), 5.0)
            self.train_metrics["mdn_loss"].update_state(mdn_loss)

        # accumulate loss into metrics

        # self.train_metrics["duration_loss"].update_state(duration_loss)
        # self.train_metrics["energy_loss"].update_state(energy_loss)
        # self.train_metrics["f0_loss"].update_state(f0_loss)
        # self.train_metrics["attention_loss"].update_state(attention_loss)
        # self.train_metrics["mel_loss_after"].update_state(mel_loss_after)
        # self.train_metrics["mel_loss_before"].update_state(mel_loss_before)

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.eval_data_loader, desc="[eval]"), 1):
            # eval one step
            characters, mels, speaker_ids, character_lengths, mel_lengths = batch

            self._eval_step(characters=characters,
                            mels=mels,
                            speaker_ids=speaker_ids,
                            character_lengths=character_lengths,
                            mel_lengths=mel_lengths)

            if eval_steps_per_epoch <= self.config["num_save_intermediate_results"]:
                # save intermedia
                self.generate_and_save_intermediate_result(batch)

        logging.info(f"(Steps: {self.steps}) Finished evaluation "
                     f"({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.eval_metrics.keys():
            logging.info(f"(Steps: {self.steps}) eval_{key} = {self.eval_metrics[key].result():.4f}.")

        # record
        self._write_to_tensorboard(self.eval_metrics, stage='eval')

        # reset
        self.reset_states_eval()

    @tf.function(experimental_relax_shapes=True,
                 input_signature=[tf.TensorSpec([None, None], dtype=tf.int32),
                                  tf.TensorSpec([None, None, 80], dtype=tf.float32),
                                  tf.TensorSpec([None], dtype=tf.int32),
                                  tf.TensorSpec([None], dtype=tf.int32),
                                  tf.TensorSpec([None], dtype=tf.int32),
                                  ])
    def _eval_step(self,
                   characters,
                   mels,
                   speaker_ids,
                   character_lengths,
                   mel_lengths):
        """Evaluate model one step."""
        masked_mel_before, masked_mel_after, \
            masked_duration_outputs, mu_sigma = self.model(
                input_ids=characters,
                speaker_ids=speaker_ids,
                durations=None,
                character_lengths=character_lengths,
                mel_lengths=mel_lengths,
                training=False)
        log_prob, mdn_loss, _ = self.mdn_loss((mels, mu_sigma, mel_lengths, character_lengths))

        self.eval_metrics["mdn_loss"].update_state(mdn_loss)
        # self.eval_metrics["duration_loss"].update_state(duration_loss)
        # self.eval_metrics["energy_loss"].update_state(energy_loss)
        # self.eval_metrics["f0_loss"].update_state(f0_loss)
        # self.eval_metrics["mel_loss_before"].update_state(mel_loss_before)
        # self.eval_metrics["mel_loss_after"].update_state(mel_loss_after)
        # self.eval_metrics["attention_loss"].update_state(attention_loss)

    def _check_log_interval(self):
        """Log to tensorboard."""
        if self.steps % self.config["log_interval_steps"] == 0:
            for metric_name in self.list_metrics_name:
                logging.info(
                    f"(Step: {self.steps}) train_{metric_name} = {self.train_metrics[metric_name].result():.4f}.")
            self._write_to_tensorboard(self.train_metrics, stage="train")

            # reset
            self.reset_states_train()

    @tf.function(experimental_relax_shapes=True,
                 input_signature=[tf.TensorSpec([None, None], dtype=tf.int32),
                                  tf.TensorSpec([None, None, 80], dtype=tf.float32),
                                  tf.TensorSpec([None], dtype=tf.int32),
                                  tf.TensorSpec([None], dtype=tf.int32),
                                  tf.TensorSpec([None], dtype=tf.int32)
                                  ])
    def predict(self, characters, mels, speaker_ids, character_lengths, mel_lengths):
        """Predict."""

        masked_mel_before, masked_mel_after, duration_pred, mu_sigma = self.model(
            input_ids=characters,
            speaker_ids=speaker_ids,
            durations=None,
            character_lengths=character_lengths,
            mel_lengths=mel_lengths,
            training=False
        )

        log_prob, _, alphas = self.mdn_loss((mels, mu_sigma, mel_lengths, character_lengths))
        align_paths = self.viterbi((log_prob, mel_lengths, character_lengths))

        return masked_mel_before, masked_mel_after, duration_pred, log_prob, alphas, align_paths

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # unpack input.
        characters, mels, speaker_ids, character_lengths, mel_lengths = batch

        # predict with tf.function.
        masked_mel_before, masked_mel_after, duration_preds, log_probs, alphas, align_paths = self.predict(
            characters,
            mels,
            speaker_ids,
            character_lengths,
            mel_lengths)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (log_prob, alpha, align_path, mel_length, character_length, character, mel) in enumerate(
                zip(log_probs, alphas, align_paths, mel_lengths, character_lengths, characters, mels),
                1):
            log_prob = log_prob.numpy()
            alpha = alpha.numpy()
            alpha[alpha < -1e6] = -1e6
            mel_length = mel_length.numpy().astype(np.float32)
            character_length = character_length.numpy().astype(np.float32)
            character = [_id_to_symbol[c] for c in character.numpy()]
            character = character[:int(character_length)]
            mel = mel.numpy()
            align_path = align_path.numpy()

            # Plot predict and ground truth mel spectrogram
            figname = os.path.join(dirname, f"{idx}.png")
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(311)
            im = ax1.imshow(log_prob.T, aspect='auto', interpolation='none', origin="lower")
            rect = plt.Rectangle((mel_length - 1.5, character_length - 1.5), 1, 1, fill=False, color="red", linewidth=3)
            ax1.add_patch(rect)
            ax1.set_title(f'log_prob_{self.steps}')
            fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
            ax2 = fig.add_subplot(312)
            im = ax2.imshow(alpha.T, aspect='auto', interpolation='none', origin="lower")
            rect = plt.Rectangle((mel_length - 1.5, character_length - 1.5), 1, 1, fill=False, color="red", linewidth=3)
            ax2.add_patch(rect)
            ax2.set_title(f'alpha_{self.steps}')
            fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax2)
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            plot_utils.plot_mel_and_alignment(
                save_folder=dirname, mel=mel, align_path=align_path,
                tokens=character, idx=idx, step=self.steps,
                mel_length=mel_length, character_length=character_length)

    def _check_train_finish(self):
        """Check training finished."""
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True

    def fit(self, train_data_loader, valid_data_loader, saved_path, resume=None):
        self.set_train_data_loader(train_data_loader)
        self.set_eval_data_loader(valid_data_loader)
        self.create_checkpoint_manager(saved_path=saved_path, max_to_keep=10000)
        if resume is not None:
            self.load_checkpoint(resume)
            logging.info(f"Successfully resumed from {resume}.")
        self.run()

    def save_checkpoint(self):
        """Save checkpoint."""
        self.ckpt.steps.assign(self.steps)
        self.ckpt.epochs.assign(self.epochs)
        self.ckp_manager.save(checkpoint_number=self.steps)
        self.model.save_weights(self.saved_path + 'model-{}.h5'.format(self.steps))

    def load_checkpoint(self, pretrained_path):
        """Load checkpoint."""
        self.ckpt.restore(pretrained_path)
        self.steps = self.ckpt.steps.numpy()
        self.epochs = self.ckpt.epochs.numpy()
        self.optimizer = self.ckpt.optimizer
        # re-assign iterations (global steps) for optimizer.
        self.optimizer.iterations.assign(tf.cast(self.steps, tf.int64))

        # load weights.
        self.model.load_weights(self.saved_path + 'model-{}.h5'.format(self.steps))


def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train FastSpeech (See detail in tensorflow_tts/bin/train-fastspeech.py)"
    )
    parser.add_argument("--train-dir", default=None, type=str,
                        help="directory including training data.")
    parser.add_argument("--valid-dir", default=None, type=str,
                        help="directory including validating data.")
    parser.add_argument("--use-norm", default=1, type=int,
                        help="usr norm-mels for train or raw.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save checkpoints.")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--resume", default=None, type=str, nargs="?",
                        help="checkpoint file path to resume training. (default=\"\")")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--mixed_precision", default=0, type=int,
                        help="using mixed precision for generator or not.")

    args = parser.parse_args()

    # set mixed precision config
    if args.mixed_precision == 1:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    args.mixed_precision = bool(args.mixed_precision)
    args.use_norm = bool(args.use_norm)

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # # check arguments
    # if args.train_dir is None:
    #     raise ValueError("Please specify --train-dir")
    # if args.dev_dir is None:
    #     raise ValueError("Please specify --valid-dir")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = tensorflow_tts.__version__
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # # get dataset
    # if config["remove_short_samples"]:
    #     mel_length_threshold = config["mel_length_threshold"]
    # else:
    #     mel_length_threshold = None

    # if config["format"] == "npy":
    #     charactor_query = "*-ids.npy"
    #     mel_query = "*-raw-feats.npy" if args.use_norm is False else "*-norm-feats.npy"
    #     duration_query = "*-durations.npy"
    #     charactor_load_fn = np.load
    #     mel_load_fn = np.load
    #     duration_load_fn = np.load
    # else:
    #     raise ValueError("Only npy are supported.")

    # define train/valid dataset
    train_dataset = LJSpeechDataset(
        root_dir=args.train_dir,
        max_mel_length=800
    ).create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"]
    )

    valid_dataset = LJSpeechDataset(
        root_dir=args.valid_dir,
        max_mel_length=1000
    ).create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"]
    )

    fastspeech = TFAlignTTS(config=AlignTTSConfig(**config["fastspeech_params"]))
    fastspeech._build()
    fastspeech.summary()

    # define trainer
    trainer = AlignTTSTrainer(config=config,
                              steps=0,
                              epochs=0,
                              is_mixed_precision=False)

    # AdamW for fastspeech
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
        decay_steps=config["optimizer_params"]["decay_steps"],
        end_learning_rate=config["optimizer_params"]["end_learning_rate"]
    )

    learning_rate_fn = WarmUp(
        initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
        decay_schedule_fn=learning_rate_fn,
        warmup_steps=int(config["train_max_steps"] * config["optimizer_params"]["warmup_proportion"])
    )

    # optimizer = tf.keras.optimizers.Adam(
    #     learning_rate=learning_rate_fn,
    #     beta_1=0.9,
    #     beta_2=0.98,
    #     epsilon=1e-9)

    optimizer = AdamWeightDecay(
        learning_rate=learning_rate_fn,
        weight_decay_rate=config["optimizer_params"]["weight_decay"],
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
        exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias']
    )

    # compile trainer
    trainer.compile(model=fastspeech,
                    optimizer=optimizer)

    # start training
    try:
        trainer.fit(train_dataset,
                    valid_dataset,
                    saved_path=os.path.join(config["outdir"], 'checkpoints/'),
                    resume=args.resume)
    except KeyboardInterrupt:
        trainer.save_checkpoint()
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    
    main()
