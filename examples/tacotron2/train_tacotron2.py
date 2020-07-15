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
"""Train Tacotron 2."""
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

import argparse
import logging
import os

import tensorflow_tts

import matplotlib.pyplot as plt
import numpy as np
import yaml

from tacotron_dataset import CharactorMelDataset
from tensorflow_tts.configs.tacotron2 import Tacotron2Config
from tensorflow_tts.models import TFTacotron2
from tensorflow_tts.optimizers import WarmUp
from tensorflow_tts.optimizers import AdamWeightDecay
from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from tensorflow_tts.utils import calculate_2d_loss, calculate_3d_loss, return_strategy
from tqdm import tqdm


os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"


class Tacotron2Trainer(Seq2SeqBasedTrainer):
    """Tacotron2 Trainer class based on Seq2SeqBasedTrainer."""

    def __init__(
        self, config, strategy, steps=0, epochs=0, is_mixed_precision=False,
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
            strategy=strategy,
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

    def compile(self, model, optimizer):
        super().compile(model, optimizer)
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        self.mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mae = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )

    def _train_step(self, batch):
        """Here we re-define _train_step because apply input_signature make
        the training progress slower on my experiment. Note that input_signature
        is apply on based_trainer by default.
        """
        if self._already_apply_input_signature is False:
            self.one_step_forward = tf.function(
                self._one_step_forward, experimental_relax_shapes=True
            )
            self.one_step_evaluate = tf.function(
                self._one_step_evaluate, experimental_relax_shapes=True
            )
            self.one_step_predict = tf.function(
                self._one_step_predict, experimental_relax_shapes=True
            )
            self._already_apply_input_signature = True

        # run one_step_forward
        self.one_step_forward(batch)

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

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
        (
            decoder_output,
            post_mel_outputs,
            stop_token_predictions,
            alignment_historys,
        ) = outputs

        mel_loss_before = calculate_3d_loss(
            batch["mel_gts"], decoder_output, loss_fn=self.mae
        )
        mel_loss_after = calculate_3d_loss(
            batch["mel_gts"], post_mel_outputs, loss_fn=self.mae
        )

        # calculate stop_loss
        stop_gts = ~tf.sequence_mask(
            batch["mel_lengths"], tf.shape(batch["mel_gts"])[1]
        )
        stop_token_loss = calculate_2d_loss(
            stop_gts, stop_token_predictions, loss_fn=self.binary_crossentropy
        )

        # calculate guided attention loss.
        if "g_attentions" in batch:
            att_mask = tf.cast(
                tf.math.not_equal(batch["g_attentions"], -1.0), tf.float32
            )
            loss_att = tf.reduce_sum(
                tf.abs(alignment_historys * batch["g_attentions"]) * att_mask,
                axis=[1, 2],
            )
            loss_att /= tf.reduce_sum(att_mask, axis=[1, 2])
        else:
            loss_att = 0

        per_example_losses = (
            stop_token_loss + mel_loss_before + mel_loss_after + loss_att
        )

        dict_metrics_losses = {
            "stop_token_loss": stop_token_loss,
            "mel_loss_before": mel_loss_before,
            "mel_loss_after": mel_loss_after,
            "guided_attention_loss": loss_att,
        }

        return per_example_losses, dict_metrics_losses

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # predict with tf.function for faster.
        (
            decoder_output,
            mel_outputs,
            stop_token_predictions,
            alignment_historys,
        ) = self.one_step_predict(batch)
        mel_gts = batch["mel_gts"]
        utt_ids = batch["utt_ids"]

        # convert to tensor.
        # here we just take a sample at first replica.
        try:
            mels_before = decoder_output.values[0].numpy()
            mels_after = mel_outputs.values[0].numpy()
            mel_gts = mel_gts.values[0].numpy()
            utt_ids = utt_ids.values[0].numpy()
            alignment_historys = alignment_historys.values[0].numpy()
        except Exception:
            mels_before = decoder_output.numpy()
            mels_after = mel_outputs.numpy()
            mel_gts = mel_gts.numpy()
            utt_ids = utt_ids.numpy()
            alignment_historys = alignment_historys.numpy()

        # check directory
        pred_output_dir = os.path.join(
            self.config["outdir"], "predictions", f"{self.steps}_steps"
        )
        os.makedirs(pred_output_dir, exist_ok=True)

        num_mels = self.config["tacotron2_params"]["n_mels"]
        items = zip(utt_ids, mel_gts, mels_before, mels_after, alignment_historys)
        for idx, (utt_id, mel_gt, mel_before, mel_after, alignment) in enumerate(items):
            mel_gt = tf.reshape(mel_gt, (-1, num_mels)).numpy()
            mel_before = tf.reshape(mel_before, (-1, num_mels)).numpy()
            mel_after = tf.reshape(mel_after, (-1, num_mels)).numpy()
            utt_id_str = utt_id.decode("utf8")

            # plot mel
            figname = os.path.join(pred_output_dir, f"{utt_id_str}.png")
            fig, axes = plt.subplots(figsize=(10, 8), nrows=3)
            for ax, data in zip(axes, [mel_gt, mel_before, mel_after]):
                im = ax.imshow(np.rot90(data), aspect="auto", interpolation="none")
                fig.colorbar(im, pad=0.02, aspect=15, orientation="vertical", ax=ax)
            axes[0].set_title(f"Target mel spectrogram ({utt_id_str})")
            axes[1].set_title(
                f"Predicted mel spectrogram before post-net @ {self.steps} steps"
            )
            axes[2].set_title(
                f"Predicted mel spectrogram after post-net @ {self.steps} steps"
            )
            plt.tight_layout()
            plt.savefig(figname, bbox_inches="tight")
            plt.close()

            # plot alignment
            figname = os.path.join(pred_output_dir, f"{utt_id_str}_alignment.png")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f"Alignment @ {self.steps} steps")
            im = ax.imshow(
                alignment, aspect="auto", origin="lower", interpolation="none"
            )
            fig.colorbar(im, aspect=15, ax=ax)
            ax.set_xlabel("Decoder timestep")
            ax.set_ylabel("Encoder timestep")
            plt.tight_layout()
            plt.savefig(figname, bbox_inches="tight")
            plt.close()


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(description="Train Tacotron 2.")
    parser.add_argument(
        "--train_dir",
        default=argparse.SUPPRESS,
        type=str,
        help="Directory containing training data.",
    )
    parser.add_argument(
        "--valid_dir",
        default=argparse.SUPPRESS,
        type=str,
        help="Directory containing validation data.",
    )
    parser.add_argument(
        "--use_norm",
        action="store_true",
        help="Whether or not to use normalized features.",
    )
    parser.add_argument(
        "--stats_path",
        default=argparse.SUPPRESS,
        type=str,
        help="Path to statistics file with mean and std values for standardization.",
    )
    parser.add_argument(
        "--outdir",
        default=argparse.SUPPRESS,
        type=str,
        help="Output directory where checkpoints and results will be saved.",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="YAML format configuration file."
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help="Checkpoint file path to resume training. (default='')",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Logging level. 0: DEBUG, 1: INFO, 2: WARN.",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Whether or not to use mixed precision training.",
    )
    args = parser.parse_args()

    STRATEGY = return_strategy()

    # load and save config
    config = yaml.load(open(args.config), Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = tensorflow_tts.__version__

    # set logger and print parameters
    fmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    log_level = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARN}
    logging.basicConfig(level=log_level[config["verbose"]], format=fmt)
    _ = [logging.info("%s = %s", key, value) for key, value in config.items()]

    # check required arguments
    missing_dirs = list(
        filter(lambda x: x not in config, ["train_dir", "valid_dir", "outdir"])
    )
    if missing_dirs:
        raise ValueError(f"{missing_dirs}.")
    if not config["use_norm"]:
        config["stats_path"] = None

    # check output directory
    os.makedirs(config["outdir"], exist_ok=True)

    with open(os.path.join(config["outdir"], "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)

    # set mixed precision config
    if config["mixed_precision"]:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    # get dataset
    train_dataset = CharactorMelDataset(
        dataset_dir=config["train_dir"],
        use_norm=config["use_norm"],
        stats_path=config["stats_path"],
        return_guided_attention=True,
        reduction_factor=config["tacotron2_params"]["reduction_factor"],
        n_mels=config["tacotron2_params"]["n_mels"],
        use_fixed_shapes=config["use_fixed_shapes"],
        mel_len_threshold=config["mel_length_threshold"],
    ).create(
        batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        training=True,
    )

    valid_dataset = CharactorMelDataset(
        dataset_dir=config["valid_dir"],
        use_norm=config["use_norm"],
        stats_path=config["stats_path"],
        return_guided_attention=True,
        reduction_factor=config["tacotron2_params"]["reduction_factor"],
        n_mels=config["tacotron2_params"]["n_mels"],
        use_fixed_shapes=False,
        mel_len_threshold=config["mel_length_threshold"],
    ).create(
        batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
        allow_cache=config["allow_cache"],
    )

    # define trainer
    trainer = Tacotron2Trainer(
        config=config,
        strategy=STRATEGY,
        steps=0,
        epochs=0,
        is_mixed_precision=args.mixed_precision,
    )

    with STRATEGY.scope():
        # define model.
        tacotron_config = Tacotron2Config(**config["tacotron2_params"])
        tacotron2 = TFTacotron2(config=tacotron_config, training=True, name="tacotron2")
        tacotron2._build()
        tacotron2.summary()

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
