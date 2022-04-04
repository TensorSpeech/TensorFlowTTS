# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team.
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
import yaml
import json

import tensorflow_tts
from TensorFlowTTS.examples.fastspeech2_libritts.fastspeech2_dataset import (
    CharactorDurationF0EnergyMelDataset,
)
from tensorflow_tts.configs import FastSpeech2Config
from tensorflow_tts.models import TFFastSpeech2
from tensorflow_tts.optimizers import AdamWeightDecay, WarmUp
from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from tensorflow_tts.utils import (
    calculate_2d_loss,
    calculate_3d_loss,
    return_strategy,
    TFGriffinLim,
)


class FastSpeech2Trainer(Seq2SeqBasedTrainer):
    """FastSpeech2 Trainer class based on FastSpeechTrainer."""

    def __init__(
        self,
        config,
        strategy,
        steps=0,
        epochs=0,
        is_mixed_precision=False,
        stats_path: str = "",
        dataset_config: str = "",
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
            strategy=strategy,
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
        self.use_griffin = config.get("use_griffin", False)
        self.griffin_lim_tf = None
        if self.use_griffin:
            logging.info(
                f"Load griff stats from {stats_path} and config from {dataset_config}"
            )
            self.griff_conf = yaml.load(open(dataset_config), Loader=yaml.Loader)
            self.prepare_grim(stats_path, self.griff_conf)

    def prepare_grim(self, stats_path, config):
        if not stats_path:
            raise KeyError("stats path need to exist")
        self.griffin_lim_tf = TFGriffinLim(stats_path, config)

    def compile(self, model, optimizer):
        super().compile(model, optimizer)
        self.mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mae = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )

    def compute_per_example_losses(self, batch, outputs):
        """Compute per example losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model
        
        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        mel_before, mel_after, duration_outputs, f0_outputs, energy_outputs = outputs

        log_duration = tf.math.log(
            tf.cast(tf.math.add(batch["duration_gts"], 1), tf.float32)
        )
        duration_loss = calculate_2d_loss(log_duration, duration_outputs, self.mse)
        f0_loss = calculate_2d_loss(batch["f0_gts"], f0_outputs, self.mse)
        energy_loss = calculate_2d_loss(batch["energy_gts"], energy_outputs, self.mse)
        mel_loss_before = calculate_3d_loss(batch["mel_gts"], mel_before, self.mae)
        mel_loss_after = calculate_3d_loss(batch["mel_gts"], mel_after, self.mae)

        per_example_losses = (
            duration_loss + f0_loss + energy_loss + mel_loss_before + mel_loss_after
        )

        dict_metrics_losses = {
            "duration_loss": duration_loss,
            "f0_loss": f0_loss,
            "energy_loss": energy_loss,
            "mel_loss_before": mel_loss_before,
            "mel_loss_after": mel_loss_after,
        }

        return per_example_losses, dict_metrics_losses

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # predict with tf.function.
        outputs = self.one_step_predict(batch)

        mels_before, mels_after, *_ = outputs
        mel_gts = batch["mel_gts"]
        utt_ids = batch["utt_ids"]

        # convert to tensor.
        # here we just take a sample at first replica.
        try:
            mels_before = mels_before.values[0].numpy()
            mels_after = mels_after.values[0].numpy()
            mel_gts = mel_gts.values[0].numpy()
            utt_ids = utt_ids.values[0].numpy()
        except Exception:
            mels_before = mels_before.numpy()
            mels_after = mels_after.numpy()
            mel_gts = mel_gts.numpy()
            utt_ids = utt_ids.numpy()

        # check directory
        if self.use_griffin:
            griff_dir_name = os.path.join(
                self.config["outdir"], f"predictions/{self.steps}_wav"
            )
            if not os.path.exists(griff_dir_name):
                os.makedirs(griff_dir_name)

        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (mel_gt, mel_before, mel_after) in enumerate(
            zip(mel_gts, mels_before, mels_after), 0
        ):

            if self.use_griffin:
                utt_id = utt_ids[idx]
                grif_before = self.griffin_lim_tf(
                    tf.reshape(mel_before, [-1, 80])[tf.newaxis, :], n_iter=32
                )
                grif_after = self.griffin_lim_tf(
                    tf.reshape(mel_after, [-1, 80])[tf.newaxis, :], n_iter=32
                )
                grif_gt = self.griffin_lim_tf(
                    tf.reshape(mel_gt, [-1, 80])[tf.newaxis, :], n_iter=32
                )
                self.griffin_lim_tf.save_wav(
                    grif_before, griff_dir_name, f"{utt_id}_before"
                )
                self.griffin_lim_tf.save_wav(
                    grif_after, griff_dir_name, f"{utt_id}_after"
                )
                self.griffin_lim_tf.save_wav(grif_gt, griff_dir_name, f"{utt_id}_gt")

            utt_id = utt_ids[idx]
            mel_gt = tf.reshape(mel_gt, (-1, 80)).numpy()  # [length, 80]
            mel_before = tf.reshape(mel_before, (-1, 80)).numpy()  # [length, 80]
            mel_after = tf.reshape(mel_after, (-1, 80)).numpy()  # [length, 80]

            # plit figure and save it
            figname = os.path.join(dirname, f"{utt_id}.png")
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            im = ax1.imshow(np.rot90(mel_gt), aspect="auto", interpolation="none")
            ax1.set_title("Target Mel-Spectrogram")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
            ax2.set_title("Predicted Mel-before-Spectrogram")
            im = ax2.imshow(np.rot90(mel_before), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)
            ax3.set_title("Predicted Mel-after-Spectrogram")
            im = ax3.imshow(np.rot90(mel_after), aspect="auto", interpolation="none")
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
        default="dump/train",
        type=str,
        help="directory including training data. ",
    )
    parser.add_argument(
        "--dev-dir",
        default="dump/valid",
        type=str,
        help="directory including development data. ",
    )
    parser.add_argument(
        "--use-norm", default=1, type=int, help="usr norm-mels for train or raw."
    )
    parser.add_argument(
        "--f0-stat", default="./dump/stats_f0.npy", type=str, help="f0-stat path.",
    )
    parser.add_argument(
        "--energy-stat",
        default="./dump/stats_energy.npy",
        type=str,
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
        default=1,
        type=int,
        help="using mixed precision for generator or not.",
    )
    parser.add_argument(
        "--dataset_config", default="preprocess/libritts_preprocess.yaml", type=str,
    )
    parser.add_argument(
        "--dataset_stats", default="dump/stats.npy", type=str,
    )
    parser.add_argument(
        "--dataset_mapping", default="dump/libritts_mapper.npy", type=str,
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        nargs="?",
        help="pretrained weights .h5 file to load weights from. Auto-skips non-matching layers",
    )
    args = parser.parse_args()

    # return strategy
    STRATEGY = return_strategy()

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

    # load speakers map from dataset map
    with open(args.dataset_mapping) as f:
        dataset_mapping = json.load(f)
        speakers_map = dataset_mapping["speakers_map"]

    # Check n_speakers matches number of speakers in speakers_map
    n_speakers = config["fastspeech2_params"]["n_speakers"]
    assert n_speakers == len(
        speakers_map
    ), f"Number of speakers in dataset does not match n_speakers in config"

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
        speakers_map=speakers_map,
    ).create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"]
        * STRATEGY.num_replicas_in_sync
        * config["gradient_accumulation_steps"],
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
        speakers_map=speakers_map,
    ).create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
    )

    # define trainer
    trainer = FastSpeech2Trainer(
        config=config,
        strategy=STRATEGY,
        steps=0,
        epochs=0,
        is_mixed_precision=args.mixed_precision,
        stats_path=args.dataset_stats,
        dataset_config=args.dataset_config,
    )

    with STRATEGY.scope():
        # define model
        fastspeech = TFFastSpeech2(
            config=FastSpeech2Config(**config["fastspeech2_params"])
        )
        fastspeech._build()
        fastspeech.summary()

        if len(args.pretrained) > 1:
            fastspeech.load_weights(args.pretrained, by_name=True, skip_mismatch=True)
            logging.info(
                f"Successfully loaded pretrained weight from {args.pretrained}."
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
                config["train_max_steps"]
                * config["optimizer_params"]["warmup_proportion"]
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

        _ = optimizer.iterations

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
