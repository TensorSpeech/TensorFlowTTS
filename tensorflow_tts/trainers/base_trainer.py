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

import tensorflow as tf
from tqdm import tqdm

from tensorflow_tts.optimizers import GradientAccumulator
from tensorflow_tts.utils import utils


class BasedTrainer(metaclass=abc.ABCMeta):
    """Customized trainer module for all models."""

    def __init__(self, steps, epochs, config):
        self.steps = steps
        self.epochs = epochs
        self.config = config
        self.finish_train = False
        self.writer = tf.summary.create_file_writer(config["outdir"])
        self.train_data_loader = None
        self.eval_data_loader = None
        self.train_metrics = None
        self.eval_metrics = None
        self.list_metrics_name = None

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

    def update_train_metrics(self, dict_metrics_losses):
        for name, value in dict_metrics_losses.items():
            self.train_metrics[name].update_state(value)

    def update_eval_metrics(self, dict_metrics_losses):
        for name, value in dict_metrics_losses.items():
            self.eval_metrics[name].update_state(value)

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
        strategy,
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
        self._is_generator_mixed_precision = is_generator_mixed_precision
        self._is_discriminator_mixed_precision = is_discriminator_mixed_precision
        self._strategy = strategy
        self._already_apply_input_signature = False
        self._generator_gradient_accumulator = GradientAccumulator()
        self._discriminator_gradient_accumulator = GradientAccumulator()
        self._generator_gradient_accumulator.reset()
        self._discriminator_gradient_accumulator.reset()

    def init_train_eval_metrics(self, list_metrics_name):
        with self._strategy.scope():
            super().init_train_eval_metrics(list_metrics_name)

    def get_n_gpus(self):
        return self._strategy.num_replicas_in_sync

    def _get_train_element_signature(self):
        return self.train_data_loader.element_spec

    def _get_eval_element_signature(self):
        return self.eval_data_loader.element_spec

    def set_gen_model(self, generator_model):
        """Set generator class model (MUST)."""
        self._generator = generator_model

    def get_gen_model(self):
        """Get generator model."""
        return self._generator

    def set_dis_model(self, discriminator_model):
        """Set discriminator class model (MUST)."""
        self._discriminator = discriminator_model

    def get_dis_model(self):
        """Get discriminator model."""
        return self._discriminator

    def set_gen_optimizer(self, generator_optimizer):
        """Set generator optimizer (MUST)."""
        self._gen_optimizer = generator_optimizer
        if self._is_generator_mixed_precision:
            self._gen_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                self._gen_optimizer, "dynamic"
            )

    def get_gen_optimizer(self):
        """Get generator optimizer."""
        return self._gen_optimizer

    def set_dis_optimizer(self, discriminator_optimizer):
        """Set discriminator optimizer (MUST)."""
        self._dis_optimizer = discriminator_optimizer
        if self._is_discriminator_mixed_precision:
            self._dis_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                self._dis_optimizer, "dynamic"
            )

    def get_dis_optimizer(self):
        """Get discriminator optimizer."""
        return self._dis_optimizer

    def compile(self, gen_model, dis_model, gen_optimizer, dis_optimizer):
        self.set_gen_model(gen_model)
        self.set_dis_model(dis_model)
        self.set_gen_optimizer(gen_optimizer)
        self.set_dis_optimizer(dis_optimizer)

    def _train_step(self, batch):
        if self._already_apply_input_signature is False:
            train_element_signature = self._get_train_element_signature()
            eval_element_signature = self._get_eval_element_signature()
            self.one_step_forward = tf.function(
                self._one_step_forward, input_signature=[train_element_signature]
            )
            self.one_step_evaluate = tf.function(
                self._one_step_evaluate, input_signature=[eval_element_signature]
            )
            self.one_step_predict = tf.function(
                self._one_step_predict, input_signature=[eval_element_signature]
            )
            self._already_apply_input_signature = True

        # run one_step_forward
        self.one_step_forward(batch)

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _one_step_forward(self, batch):
        per_replica_losses = self._strategy.run(
            self._one_step_forward_per_replica, args=(batch,)
        )
        return self._strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )

    @abc.abstractmethod
    def compute_per_example_generator_losses(self, batch, outputs):
        """Compute per example generator losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model
        
        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        per_example_losses = 0.0
        dict_metrics_losses = {}
        return per_example_losses, dict_metrics_losses

    @abc.abstractmethod
    def compute_per_example_discriminator_losses(self, batch, gen_outputs):
        """Compute per example discriminator losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model
        
        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        per_example_losses = 0.0
        dict_metrics_losses = {}
        return per_example_losses, dict_metrics_losses

    def _calculate_generator_gradient_per_batch(self, batch):
        outputs = self._generator(**batch, training=True)
        (
            per_example_losses,
            dict_metrics_losses,
        ) = self.compute_per_example_generator_losses(batch, outputs)
        per_replica_gen_losses = tf.nn.compute_average_loss(
            per_example_losses,
            global_batch_size=self.config["batch_size"]
            * self.get_n_gpus()
            * self.config["gradient_accumulation_steps"],
        )

        if self._is_generator_mixed_precision:
            scaled_per_replica_gen_losses = self._gen_optimizer.get_scaled_loss(
                per_replica_gen_losses
            )

        if self._is_generator_mixed_precision:
            scaled_gradients = tf.gradients(
                scaled_per_replica_gen_losses, self._generator.trainable_variables
            )
            gradients = self._gen_optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tf.gradients(
                per_replica_gen_losses, self._generator.trainable_variables
            )

        # gradient accumulate for generator here
        if self.config["gradient_accumulation_steps"] > 1:
            self._generator_gradient_accumulator(gradients)

        # accumulate loss into metrics
        self.update_train_metrics(dict_metrics_losses)

        if self.config["gradient_accumulation_steps"] == 1:
            return gradients, per_replica_gen_losses
        else:
            return per_replica_gen_losses

    def _calculate_discriminator_gradient_per_batch(self, batch):
        (
            per_example_losses,
            dict_metrics_losses,
        ) = self.compute_per_example_discriminator_losses(
            batch, self._generator(**batch, training=True)
        )

        per_replica_dis_losses = tf.nn.compute_average_loss(
            per_example_losses,
            global_batch_size=self.config["batch_size"]
            * self.get_n_gpus()
            * self.config["gradient_accumulation_steps"],
        )

        if self._is_discriminator_mixed_precision:
            scaled_per_replica_dis_losses = self._dis_optimizer.get_scaled_loss(
                per_replica_dis_losses
            )

        if self._is_discriminator_mixed_precision:
            scaled_gradients = tf.gradients(
                scaled_per_replica_dis_losses,
                self._discriminator.trainable_variables,
            )
            gradients = self._dis_optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tf.gradients(
                per_replica_dis_losses, self._discriminator.trainable_variables
            )

        # accumulate loss into metrics
        self.update_train_metrics(dict_metrics_losses)

        # gradient accumulate for discriminator here
        if self.config["gradient_accumulation_steps"] > 1:
            self._discriminator_gradient_accumulator(gradients)

        if self.config["gradient_accumulation_steps"] == 1:
            return gradients, per_replica_dis_losses
        else:
            return per_replica_dis_losses


    def _one_step_forward_per_replica(self, batch):
        per_replica_gen_losses = 0.0
        per_replica_dis_losses = 0.0

        if self.config["gradient_accumulation_steps"] == 1:
            (
                gradients,
                per_replica_gen_losses,
            ) = self._calculate_generator_gradient_per_batch(batch)
            self._gen_optimizer.apply_gradients(
                zip(gradients, self._generator.trainable_variables)
            )
        else:
            # gradient acummulation here.
            for i in tf.range(self.config["gradient_accumulation_steps"]):
                reduced_batch = {
                    k: v[
                        i
                        * self.config["batch_size"] : (i + 1)
                        * self.config["batch_size"]
                    ]
                    for k, v in batch.items()
                }

                # run 1 step accumulate
                reduced_batch_losses = self._calculate_generator_gradient_per_batch(
                    reduced_batch
                )

                # sum per_replica_losses
                per_replica_gen_losses += reduced_batch_losses

            gradients = self._generator_gradient_accumulator.gradients
            self._gen_optimizer.apply_gradients(
                zip(gradients, self._generator.trainable_variables)
            )
            self._generator_gradient_accumulator.reset()

        # one step discriminator
        # recompute y_hat after 1 step generator for discriminator training.
        if self.steps >= self.config["discriminator_train_start_steps"]:
            if self.config["gradient_accumulation_steps"] == 1:
                (
                    gradients,
                    per_replica_dis_losses,
                ) = self._calculate_discriminator_gradient_per_batch(batch)
                self._dis_optimizer.apply_gradients(
                    zip(gradients, self._discriminator.trainable_variables)
                )
            else:
                # gradient acummulation here.
                for i in tf.range(self.config["gradient_accumulation_steps"]):
                    reduced_batch = {
                        k: v[
                            i
                            * self.config["batch_size"] : (i + 1)
                            * self.config["batch_size"]
                        ]
                        for k, v in batch.items()
                    }

                    # run 1 step accumulate
                    reduced_batch_losses = (
                        self._calculate_discriminator_gradient_per_batch(reduced_batch)
                    )

                    # sum per_replica_losses
                    per_replica_dis_losses += reduced_batch_losses

                gradients = self._discriminator_gradient_accumulator.gradients
                self._dis_optimizer.apply_gradients(
                    zip(gradients, self._discriminator.trainable_variables)
                )
                self._discriminator_gradient_accumulator.reset()

        return per_replica_gen_losses + per_replica_dis_losses

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.eval_data_loader, desc="[eval]"), 1
        ):
            # eval one step
            self.one_step_evaluate(batch)

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

    def _one_step_evaluate_per_replica(self, batch):
        ################################################
        # one step generator.
        outputs = self._generator(**batch, training=False)
        _, dict_metrics_losses = self.compute_per_example_generator_losses(
            batch, outputs
        )

        # accumulate loss into metrics
        self.update_eval_metrics(dict_metrics_losses)

        ################################################
        # one step discriminator
        if self.steps >= self.config["discriminator_train_start_steps"]:
            _, dict_metrics_losses = self.compute_per_example_discriminator_losses(
                batch, outputs
            )

            # accumulate loss into metrics
            self.update_eval_metrics(dict_metrics_losses)

    ################################################

    def _one_step_evaluate(self, batch):
        self._strategy.run(self._one_step_evaluate_per_replica, args=(batch,))

    def _one_step_predict_per_replica(self, batch):
        outputs = self._generator(**batch, training=False)
        return outputs

    def _one_step_predict(self, batch):
        outputs = self._strategy.run(self._one_step_predict_per_replica, args=(batch,))
        return outputs

    @abc.abstractmethod
    def generate_and_save_intermediate_result(self, batch):
        return

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
        utils.save_weights(
            self._generator,
            self.saved_path + "generator-{}.h5".format(self.steps)
        )
        utils.save_weights(
            self._discriminator,
            self.saved_path + "discriminator-{}.h5".format(self.steps)
        )

    def load_checkpoint(self, pretrained_path):
        """Load checkpoint."""
        self.ckpt.restore(pretrained_path)
        self.steps = self.ckpt.steps.numpy()
        self.epochs = self.ckpt.epochs.numpy()
        self._gen_optimizer = self.ckpt.gen_optimizer
        # re-assign iterations (global steps) for gen_optimizer.
        self._gen_optimizer.iterations.assign(tf.cast(self.steps, tf.int64))
        # re-assign iterations (global steps) for dis_optimizer.
        try:
            discriminator_train_start_steps = self.config[
                "discriminator_train_start_steps"
            ]
            discriminator_train_start_steps = tf.math.maximum(
                0, self.steps - discriminator_train_start_steps 
            )
        except Exception:
            discriminator_train_start_steps = self.steps
        self._dis_optimizer = self.ckpt.dis_optimizer
        self._dis_optimizer.iterations.assign(
            tf.cast(discriminator_train_start_steps, tf.int64)
        )

        # load weights.
        utils.load_weights(
            self._generator,
            self.saved_path + "generator-{}.h5".format(self.steps)
        )
        utils.load_weights(
            self._discriminator,
            self.saved_path + "discriminator-{}.h5".format(self.steps)
        )

    def _check_train_finish(self):
        """Check training finished."""
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True

        if (
            self.steps != 0
            and self.steps == self.config["discriminator_train_start_steps"]
        ):
            self.finish_train = True
            logging.info(
                f"Finished training only generator at {self.steps}steps, pls resume and continue training."
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

    def fit(self, train_data_loader, valid_data_loader, saved_path, resume=None):
        self.set_train_data_loader(train_data_loader)
        self.set_eval_data_loader(valid_data_loader)
        self.train_data_loader = self._strategy.experimental_distribute_dataset(
            self.train_data_loader
        )
        self.eval_data_loader = self._strategy.experimental_distribute_dataset(
            self.eval_data_loader
        )
        with self._strategy.scope():
            self.create_checkpoint_manager(saved_path=saved_path, max_to_keep=10000)
            if len(resume) > 1:
                self.load_checkpoint(resume)
                logging.info(f"Successfully resumed from {resume}.")
        self.run()


class Seq2SeqBasedTrainer(BasedTrainer, metaclass=abc.ABCMeta):
    """Customized trainer module for Seq2Seq TTS training (Tacotron, FastSpeech)."""

    def __init__(
        self, steps, epochs, config, strategy, is_mixed_precision=False,
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            strategy (tf.distribute): Strategy for distributed training.
            is_mixed_precision (bool): Use mixed_precision training or not.

        """
        super().__init__(steps, epochs, config)
        self._is_mixed_precision = is_mixed_precision
        self._strategy = strategy
        self._model = None
        self._optimizer = None
        self._trainable_variables = None

        # check if we already apply input_signature for train_step.
        self._already_apply_input_signature = False

        # create gradient accumulator
        self._gradient_accumulator = GradientAccumulator()
        self._gradient_accumulator.reset()

    def init_train_eval_metrics(self, list_metrics_name):
        with self._strategy.scope():
            super().init_train_eval_metrics(list_metrics_name)

    def set_model(self, model):
        """Set generator class model (MUST)."""
        self._model = model

    def get_model(self):
        """Get generator model."""
        return self._model

    def set_optimizer(self, optimizer):
        """Set optimizer (MUST)."""
        self._optimizer = optimizer
        if self._is_mixed_precision:
            self._optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                self._optimizer, "dynamic"
            )

    def get_optimizer(self):
        """Get optimizer."""
        return self._optimizer

    def get_n_gpus(self):
        return self._strategy.num_replicas_in_sync

    def compile(self, model, optimizer):
        self.set_model(model)
        self.set_optimizer(optimizer)
        self._trainable_variables = self._train_vars()

    def _train_vars(self):
        if self.config["var_train_expr"]:
            list_train_var = self.config["var_train_expr"].split("|")
            return [
                v
                for v in self._model.trainable_variables
                if self._check_string_exist(list_train_var, v.name)
            ]
        return self._model.trainable_variables

    def _check_string_exist(self, list_string, inp_string):
        for string in list_string:
            if string in inp_string:
                return True
        return False

    def _get_train_element_signature(self):
        return self.train_data_loader.element_spec

    def _get_eval_element_signature(self):
        return self.eval_data_loader.element_spec

    def _train_step(self, batch):
        if self._already_apply_input_signature is False:
            train_element_signature = self._get_train_element_signature()
            eval_element_signature = self._get_eval_element_signature()
            self.one_step_forward = tf.function(
                self._one_step_forward, input_signature=[train_element_signature]
            )
            self.one_step_evaluate = tf.function(
                self._one_step_evaluate, input_signature=[eval_element_signature]
            )
            self.one_step_predict = tf.function(
                self._one_step_predict, input_signature=[eval_element_signature]
            )
            self._already_apply_input_signature = True

        # run one_step_forward
        self.one_step_forward(batch)

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _one_step_forward(self, batch):
        per_replica_losses = self._strategy.run(
            self._one_step_forward_per_replica, args=(batch,)
        )
        return self._strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )

    def _calculate_gradient_per_batch(self, batch):
        outputs = self._model(**batch, training=True)
        per_example_losses, dict_metrics_losses = self.compute_per_example_losses(
            batch, outputs
        )
        per_replica_losses = tf.nn.compute_average_loss(
            per_example_losses,
            global_batch_size=self.config["batch_size"]
            * self.get_n_gpus()
            * self.config["gradient_accumulation_steps"],
        )

        if self._is_mixed_precision:
            scaled_per_replica_losses = self._optimizer.get_scaled_loss(
                per_replica_losses
            )

        if self._is_mixed_precision:
            scaled_gradients = tf.gradients(
                scaled_per_replica_losses, self._trainable_variables
            )
            gradients = self._optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tf.gradients(per_replica_losses, self._trainable_variables)

        # gradient accumulate here
        if self.config["gradient_accumulation_steps"] > 1:
            self._gradient_accumulator(gradients)

        # accumulate loss into metrics
        self.update_train_metrics(dict_metrics_losses)

        if self.config["gradient_accumulation_steps"] == 1:
            return gradients, per_replica_losses
        else:
            return per_replica_losses

    def _one_step_forward_per_replica(self, batch):
        if self.config["gradient_accumulation_steps"] == 1:
            gradients, per_replica_losses = self._calculate_gradient_per_batch(batch)
            self._optimizer.apply_gradients(
                zip(gradients, self._trainable_variables), 1.0
            )
        else:
            # gradient acummulation here.
            per_replica_losses = 0.0
            for i in tf.range(self.config["gradient_accumulation_steps"]):
                reduced_batch = {
                    k: v[
                        i
                        * self.config["batch_size"] : (i + 1)
                        * self.config["batch_size"]
                    ]
                    for k, v in batch.items()
                }

                # run 1 step accumulate
                reduced_batch_losses = self._calculate_gradient_per_batch(reduced_batch)

                # sum per_replica_losses
                per_replica_losses += reduced_batch_losses

            gradients = self._gradient_accumulator.gradients
            self._optimizer.apply_gradients(
                zip(gradients, self._trainable_variables), 1.0
            )
            self._gradient_accumulator.reset()

        return per_replica_losses


    @abc.abstractmethod
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
        per_example_losses = 0.0
        dict_metrics_losses = {}
        return per_example_losses, dict_metrics_losses

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.eval_data_loader, desc="[eval]"), 1
        ):
            # eval one step
            self.one_step_evaluate(batch)

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

    def _one_step_evaluate_per_replica(self, batch):
        outputs = self._model(**batch, training=False)
        _, dict_metrics_losses = self.compute_per_example_losses(batch, outputs)

        self.update_eval_metrics(dict_metrics_losses)

    def _one_step_evaluate(self, batch):
        self._strategy.run(self._one_step_evaluate_per_replica, args=(batch,))

    def _one_step_predict_per_replica(self, batch):
        outputs = self._model(**batch, training=False)
        return outputs

    def _one_step_predict(self, batch):
        outputs = self._strategy.run(self._one_step_predict_per_replica, args=(batch,))
        return outputs

    @abc.abstractmethod
    def generate_and_save_intermediate_result(self, batch):
        return

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
        utils.save_weights(
            self._model,
            self.saved_path + "model-{}.h5".format(self.steps)
        )

    def load_checkpoint(self, pretrained_path):
        """Load checkpoint."""
        self.ckpt.restore(pretrained_path)
        self.steps = self.ckpt.steps.numpy()
        self.epochs = self.ckpt.epochs.numpy()
        self._optimizer = self.ckpt.optimizer
        # re-assign iterations (global steps) for optimizer.
        self._optimizer.iterations.assign(tf.cast(self.steps, tf.int64))

        # load weights.
        utils.load_weights(
            self._model,
            self.saved_path + "model-{}.h5".format(self.steps)
        )

    def _check_train_finish(self):
        """Check training finished."""
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True

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

    def fit(self, train_data_loader, valid_data_loader, saved_path, resume=None):
        self.set_train_data_loader(train_data_loader)
        self.set_eval_data_loader(valid_data_loader)
        self.train_data_loader = self._strategy.experimental_distribute_dataset(
            self.train_data_loader
        )
        self.eval_data_loader = self._strategy.experimental_distribute_dataset(
            self.eval_data_loader
        )
        with self._strategy.scope():
            self.create_checkpoint_manager(saved_path=saved_path, max_to_keep=10000)
            if len(resume) > 1:
                self.load_checkpoint(resume)
                logging.info(f"Successfully resumed from {resume}.")
        self.run()
