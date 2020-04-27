# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen Quan Anh
#  MIT License (https://opensource.org/licenses/MIT)

"""Train FastSpeech."""

import argparse
import logging
import os
import sys

import numpy as np
import soundfile as sf
import tensorflow as tf
import yaml

import tensorflow_tts

from tqdm import tqdm

from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from tensorflow_tts.datasets import CharactorDurationMelDataset

from tensorflow_tts.models import TFFastSpeech

import tensorflow_tts.configs.fastspeech as FASTSPEECH_CONFIG


class FastSpeechTrainer(Seq2SeqBasedTrainer):
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
        super(FastSpeechTrainer, self).__init__(steps=steps,
                                                epochs=epochs,
                                                config=config,
                                                is_mixed_precision=is_mixed_precision)
        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "duration_loss",
            "mel_loss"
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

    def _train_step(self, batch):
        """Train model one step."""
        charactor, duration, mel = batch
        self._one_step_fastspeech(charactor, duration, mel)

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    @tf.function(experimental_relax_shapes=True,
                 input_signature=[tf.TensorSpec([None, None], dtype=tf.int32),
                                  tf.TensorSpec([None, None], dtype=tf.int32),
                                  tf.TensorSpec([None, None, 80], dtype=tf.float32)])
    def _one_step_fastspeech(self, charactor, duration, mel):
        with tf.GradientTape() as tape:
            masked_mel_outputs, masked_duration_outputs = self.model(
                charactor,
                attention_mask=tf.math.not_equal(charactor, 0),
                speaker_ids=tf.zeros(shape=[tf.shape(mel)[0]]),
                duration_gts=duration,
                training=True
            )
            duration_loss = self.mse_log(duration, masked_duration_outputs)
            mel_loss = self.mse(mel, masked_mel_outputs)
            loss = duration_loss + mel_loss

            if self.is_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # accumulate loss into metrics
        self.train_metrics["duration_loss"].update_state(self.mae(duration, masked_duration_outputs))
        self.train_metrics["mel_loss"].update_state(mel_loss)

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.eval_data_loader, desc="[eval]"), 1):
            # eval one step
            charactor, duration, mel = batch
            self._eval_step(charactor, duration, mel)

            if eval_steps_per_epoch <= self.config["num_save_intermediate_results"]:
                # save intermedia
                self.generate_and_save_intermediate_result(batch, eval_steps_per_epoch)

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
                                  tf.TensorSpec([None, None], dtype=tf.int32),
                                  tf.TensorSpec([None, None, 80], dtype=tf.float32)])
    def _eval_step(self, charactor, duration, mel):
        """Evaluate model one step."""
        masked_mel_outputs, masked_duration_outputs = self.model(
            charactor,
            attention_mask=tf.math.not_equal(charactor, 0),
            speaker_ids=tf.zeros(shape=[tf.shape(charactor)[0]]),
            duration_gts=duration,
            training=True
        )
        duration_loss = self.mae(duration, masked_duration_outputs)
        mel_loss = self.mse(mel, masked_mel_outputs)

        # accumulate loss into metrics
        self.eval_metrics["duration_loss"].update_state(duration_loss)
        self.eval_metrics["mel_loss"].update_state(mel_loss)

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
                                  tf.TensorSpec([None, None], dtype=tf.int32),
                                  tf.TensorSpec([None, None, 80], dtype=tf.float32)])
    def predict(self, charactor, duration, mel):
        """Predict."""
        masked_mel_outputs, _ = self.model(
            charactor,
            attention_mask=tf.math.not_equal(charactor, 0),
            speaker_ids=tf.zeros(shape=[tf.shape(charactor)[0]]),
            duration_gts=duration,
            training=True
        )
        return masked_mel_outputs

    def generate_and_save_intermediate_result(self, batch, idx):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # unpack input.
        charactor, duration, mel = batch

        # predict with tf.function.
        mel_predict = self.predict(charactor, duration, mel)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for _, (mel_gt, mel_pred) in enumerate(zip(mel, mel_predict), 1):
            mel_gt = tf.reshape(mel_gt, (-1, 80)).numpy()  # [length, 80]
            mel_pred = tf.reshape(mel_pred, (-1, 80)).numpy()  # [length, 80]

            # plit figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            im = ax1.imshow(np.rot90(mel_gt), aspect='auto', interpolation='none')
            ax1.set_title('Target Mel-Spectrogram')
            fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
            ax2.set_title('Predicted Mel-Spectrogram')
            im = ax2.imshow(np.rot90(mel_pred), aspect='auto', interpolation='none')
            fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax2)
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

    def _check_train_finish(self):
        """Check training finished."""
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True

    def fit(self, train_data_loader, valid_data_loader, saved_path):
        self.set_train_data_loader(train_data_loader)
        self.set_eval_data_loader(valid_data_loader)
        self.create_checkpoint_manager(saved_path=saved_path, max_to_keep=10000)
        self.run()


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train FastSpeech (See detail in tensorflow_tts/bin/train-fastspeech.py)"
    )
    parser.add_argument("--train-dir", default=None, type=str,
                        help="directory including training data. "
                             "you need to specify either train-*-scp or train-dumpdir.")
    parser.add_argument("--dev-dir", default=None, type=str,
                        help="directory including development data. "
                             "you need to specify either dev-*-scp or dev-dumpdir.")
    parser.add_argument("--use-norm", default=False, type=bool,
                        help="directory including development data. "
                             "you need to specify either dev-*-scp or dev-dumpdir.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save checkpoints.")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--pretrain", default="", type=str, nargs="?",
                        help="checkpoint file path to load pretrained params. (default=\"\")")
    parser.add_argument("--resume", default="", type=str, nargs="?",
                        help="checkpoint file path to resume training. (default=\"\")")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--rank", "--local_rank", default=0, type=int,
                        help="rank for distributed training. no need to explictly specify.")
    parser.add_argument("--mixed_precision", default=0, type=int,
                        help="using mixed precision for generator or not.")
    args = parser.parse_args()

    # set mixed precision config
    if args.mixed_precision == 1:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    args.mixed_precision = bool(args.mixed_precision)

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
        mel_length_threshold = config["batch_max_steps"]
    else:
        mel_length_threshold = None

    if config["format"] == "npy":
        charactor_query = "*-ids.npy"
        mel_query = "*-raw-feats.npy" if args.use_norm is False else "*-norm-feats.npy"
        duration_query = "*-durations.npy"
        charactor_load_fn = np.load
        mel_load_fn = np.load
        duration_load_fn = np.load
    else:
        raise ValueError("Only npy are supported.")

    # define train/valid dataset
    train_dataset = CharactorDurationMelDataset(
        root_dir=args.train_dir,
        charactor_query=charactor_query,
        mel_query=mel_query,
        duration_query=duration_query,
        charactor_load_fn=charactor_load_fn,
        mel_load_fn=mel_load_fn,
        duration_load_fn=duration_load_fn,
        mel_length_threshold=32,
        return_utt_id=False
    ).create(
        is_shuffle=False,
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"]
    )

    valid_dataset = CharactorDurationMelDataset(
        root_dir=args.dev_dir,
        charactor_query=charactor_query,
        mel_query=mel_query,
        duration_query=duration_query,
        charactor_load_fn=charactor_load_fn,
        mel_load_fn=mel_load_fn,
        duration_load_fn=duration_load_fn,
        mel_length_threshold=None,
        return_utt_id=False
    ).create(
        is_shuffle=True,
        allow_cache=config["allow_cache"],
        batch_size=1
    )

    fastspeech = TFFastSpeech(config=FASTSPEECH_CONFIG.FastSpeechConfig(**config["fastspeech_params"]))
    fastspeech.summary()

    # define trainer
    trainer = FastSpeechTrainer(config=config,
                                steps=0,
                                epochs=0,
                                is_mixed_precision=False)

    # compile trainer
    trainer.compile(model=fastspeech,
                    optimizer=tf.keras.optimizers.Adam(lr=0.001))

    # load pretrained
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # start training
    try:
        trainer.fit(train_dataset,
                    valid_dataset,
                    saved_path=config["outdir"] + '/checkpoints/')
    except KeyboardInterrupt:
        trainer.save_checkpoint()
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
