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
"""Based Trainer."""

import abc
import logging
import os
from tqdm import tqdm

import tensorflow as tf


class BasedTrainer(metaclass=abc.ABCMeta):
    """Customized trainer module for all models."""

    def __init__(self, steps, epochs, config):
        self.steps = steps
        self.epochs = epochs
        self.config = config
        self.finish_train = False
        self.writer = tf.summary.create_file_writer(config["outdir"])
        self.train_data_loader = None

    def set_train_data_loader(self, train_dataset):
        """Set train data loader (MUST)."""
        self.train_data_loader = train_dataset

    def get_train_data_loader(self):
        """Get train data loader."""
        return self.train_data_loader

    def set_eval_data_loader(self, eval_dataset):
        """Set eval data loader (MUST)."""
        self.eval_data_loader = eval_dataset

    def get_eval_data_loader(self):
        """Get eval data loader."""
        return self.eval_data_loader

    @abc.abstractmethod
    def compile(self):
        pass

    @abc.abstractmethod
    def create_checkpoint_manager(self, saved_path=None, max_to_keep=10):
        """Create checkpoint management."""
        pass

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            self._train_epoch()

            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finish training.")

    @abc.abstractmethod
    def save_checkpoint(self):
        """Save checkpoint."""
        pass

    @abc.abstractmethod
    def load_checkpoint(self, pretrained_path):
        """Load checkpoint."""
        pass

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.train_data_loader, 1):
            # one step training
            self._train_step(batch)

            # check interval
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()

            # check wheter training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

    @abc.abstractmethod
    def _eval_epoch(self):
        """One epoch evaluation."""
        pass

    @abc.abstractmethod
    def _train_step(self, batch):
        """One step training."""
        pass

    @abc.abstractmethod
    def _eval_step(self, batch):
        """One eval step."""
        pass

    @abc.abstractmethod
    def _check_log_interval(self):
        """Save log interval."""
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    def _check_eval_interval(self):
        """Evaluation interval step."""
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_save_interval(self):
        """Save interval checkpoint."""
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint()
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        pass

    def _write_to_tensorboard(self, list_metrics, stage="train"):
        """Write variables to tensorboard."""
        with self.writer.as_default():
            for key, value in list_metrics.items():
                tf.summary.scalar(stage + "/" + key, value.result(), step=self.steps)
                self.writer.flush()


class GanBasedTrainer(BasedTrainer):
    """Customized trainer module for GAN TTS training (MelGAN, GAN-TTS, ParallelWaveGAN)."""

    def __init__(
        self,
        steps,
        epochs,
        config,
        is_generator_mixed_precision=False,
        is_discriminator_mixed_precision=False,
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.

        """
        super().__init__(steps, epochs, config)
        self.is_generator_mixed_precision = is_generator_mixed_precision
        self.is_discriminator_mixed_precision = is_discriminator_mixed_precision

    def set_gen_model(self, generator_model):
        """Set generator class model (MUST)."""
        self.generator = generator_model

    def get_gen_model(self):
        """Get generator model."""
        return self.generator

    def set_dis_model(self, discriminator_model):
        """Set discriminator class model (MUST)."""
        self.discriminator = discriminator_model

    def get_dis_model(self):
        """Get discriminator model."""
        return self.discriminator

    def set_gen_optimizer(self, generator_optimizer):
        """Set generator optimizer (MUST)."""
        self.gen_optimizer = generator_optimizer
        if self.is_generator_mixed_precision:
            self.gen_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                self.gen_optimizer, "dynamic"
            )

    def get_gen_optimizer(self):
        """Get generator optimizer."""
        return self.gen_optimizer

    def set_dis_optimizer(self, discriminator_optimizer):
        """Set discriminator optimizer (MUST)."""
        self.dis_optimizer = discriminator_optimizer
        if self.is_discriminator_mixed_precision:
            self.dis_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                self.dis_optimizer, "dynamic"
            )

    def get_dis_optimizer(self):
        """Get discriminator optimizer."""
        return self.dis_optimizer

    def compile(self, gen_model, dis_model, gen_optimizer, dis_optimizer):
        self.set_gen_model(gen_model)
        self.set_dis_model(dis_model)
        self.set_gen_optimizer(gen_optimizer)
        self.set_dis_optimizer(dis_optimizer)

    def create_checkpoint_manager(self, saved_path=None, max_to_keep=10):
        """Create checkpoint management."""
        if saved_path is None:
            saved_path = self.config["outdir"] + "/checkpoints/"

        os.makedirs(saved_path, exist_ok=True)

        self.saved_path = saved_path
        self.ckpt = tf.train.Checkpoint(
            steps=tf.Variable(1),
            epochs=tf.Variable(1),
            gen_optimizer=self.get_gen_optimizer(),
            dis_optimizer=self.get_dis_optimizer(),
        )
        self.ckp_manager = tf.train.CheckpointManager(
            self.ckpt, saved_path, max_to_keep=max_to_keep
        )

    def save_checkpoint(self):
        """Save checkpoint."""
        self.ckpt.steps.assign(self.steps)
        self.ckpt.epochs.assign(self.epochs)
        self.ckp_manager.save(checkpoint_number=self.steps)
        self.generator.save_weights(
            self.saved_path + "generator-{}.h5".format(self.steps)
        )
        self.discriminator.save_weights(
            self.saved_path + "discriminator-{}.h5".format(self.steps)
        )

    def load_checkpoint(self, pretrained_path):
        """Load checkpoint."""
        self.ckpt.restore(pretrained_path)
        self.steps = self.ckpt.steps.numpy()
        self.epochs = self.ckpt.epochs.numpy()
        self.gen_optimizer = self.ckpt.gen_optimizer
        # re-assign iterations (global steps) for gen_optimizer.
        self.gen_optimizer.iterations.assign(tf.cast(self.steps, tf.int64))
        # re-assign iterations (global steps) for dis_optimizer.
        try:
            discriminator_train_start_steps = self.config[
                "discriminator_train_start_steps"
            ]
            discriminator_train_start_steps = tf.math.maximum(
                0, discriminator_train_start_steps - self.steps
            )
        except Exception:
            discriminator_train_start_steps = self.steps
        self.dis_optimizer = self.ckpt.dis_optimizer
        self.dis_optimizer.iterations.assign(
            tf.cast(discriminator_train_start_steps, tf.int64)
        )

        # load weights.
        self.generator.load_weights(
            self.saved_path + "generator-{}.h5".format(self.steps)
        )
        self.discriminator.load_weights(
            self.saved_path + "discriminator-{}.h5".format(self.steps)
        )


class Seq2SeqBasedTrainer(BasedTrainer):
    """Customized trainer module for Seq2Seq TTS training (Tacotron, FastSpeech)."""

    def __init__(
        self, steps, epochs, config, is_mixed_precision=False,
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.

        """
        super().__init__(steps, epochs, config)
        self.is_mixed_precision = is_mixed_precision

    def set_model(self, model):
        """Set generator class model (MUST)."""
        self.model = model

    def get_model(self):
        """Get generator model."""
        return self.model

    def set_optimizer(self, optimizer):
        """Set optimizer (MUST)."""
        self.optimizer = optimizer
        if self.is_mixed_precision:
            self.optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                self.optimizer, "dynamic"
            )

    def get_optimizer(self):
        """Get optimizer."""
        return self.optimizer

    def compile(self, model, optimizer):
        self.set_model(model)
        self.set_optimizer(optimizer)

    def create_checkpoint_manager(self, saved_path=None, max_to_keep=10):
        """Create checkpoint management."""
        if saved_path is None:
            saved_path = self.config["outdir"] + "/checkpoints/"

        os.makedirs(saved_path, exist_ok=True)

        self.saved_path = saved_path
        self.ckpt = tf.train.Checkpoint(
            steps=tf.Variable(1), epochs=tf.Variable(1), optimizer=self.get_optimizer()
        )
        self.ckp_manager = tf.train.CheckpointManager(
            self.ckpt, saved_path, max_to_keep=max_to_keep
        )

    def save_checkpoint(self):
        """Save checkpoint."""
        self.ckpt.steps.assign(self.steps)
        self.ckpt.epochs.assign(self.epochs)
        self.ckp_manager.save(checkpoint_number=self.steps)
        self.model.save_weights(self.saved_path + "model-{}.h5".format(self.steps))

    def load_checkpoint(self, pretrained_path):
        """Load checkpoint."""
        self.ckpt.restore(pretrained_path)
        self.steps = self.ckpt.steps.numpy()
        self.epochs = self.ckpt.epochs.numpy()
        self.optimizer = self.ckpt.optimizer
        # re-assign iterations (global steps) for optimizer.
        self.optimizer.iterations.assign(tf.cast(self.steps, tf.int64))

        # load weights.
        self.model.load_weights(self.saved_path + "model-{}.h5".format(self.steps))
