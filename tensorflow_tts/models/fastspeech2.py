# -*- coding: utf-8 -*-
# Copyright 2020 The FastSpeech2 Authors and Minh Nguyen (@dathudeptrai)
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
"""Tensorflow Model modules for FastSpeech2."""

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import math_ops

from tensorflow_tts.models.fastspeech import TFFastSpeech
from tensorflow_tts.models.fastspeech import get_initializer


class TFFastSpeechVariantPredictor(tf.keras.layers.Layer):
    """FastSpeech duration predictor module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_layers = []
        for i in range(config.num_duration_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(
                    config.f0_energy_predictor_filters,
                    config.f0_energy_predictor_kernel_sizes,
                    padding='same',
                    name='conv_._{}'.format(i)
                )
            )
            self.conv_layers.append(
                tf.keras.layers.Activation(tf.nn.relu)
            )
            self.conv_layers.append(
                tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm_._{}".format(i))
            )
            self.conv_layers.append(
                tf.keras.layers.Dropout(config.f0_energy_predictor_dropout_probs)
            )
        self.conv_layers_sequence = tf.keras.Sequential(self.conv_layers)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        """Call logic."""
        encoder_hidden_states, attention_mask = inputs
        attention_mask = tf.cast(tf.expand_dims(attention_mask, 2), tf.float32)

        # mask encoder hidden states
        masked_encoder_hidden_states = encoder_hidden_states

        # pass though first layer
        outputs = self.conv_layers_sequence(masked_encoder_hidden_states)
        outputs = self.output_layer(outputs)
        masked_outputs = outputs

        outputs = tf.squeeze(masked_outputs, -1)
        return outputs


class TFFastSpeech2(TFFastSpeech):
    """TF Fastspeech module."""

    def __init__(self, config, **kwargs):
        """Init layers for fastspeech."""
        super().__init__(config, **kwargs)
        self.f0_predictor = TFFastSpeechVariantPredictor(config, name="f0_predictor")
        self.energy_predictor = TFFastSpeechVariantPredictor(config, name="energy_predictor")
        self.duration_predictor = TFFastSpeechVariantPredictor(config, name='duration_predictor')

        # define f0_embeddings and energy_embeddings
        self.f0_embeddings = tf.keras.layers.Embedding(
            config.max_f0_embeddings + 1,  # +1 for padding.
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="f0_embeddings",
        )
        self.energy_embeddings = tf.keras.layers.Embedding(
            config.max_energy_embeddings + 1,  # +1 for padding
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="energy_embeddings",
        )

    def _build(self):
        """Dummy input for building model."""
        # fake inputs
        input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        attention_mask = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        duration_gts = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
        f0_gts = tf.convert_to_tensor([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]], tf.int32)
        energy_gts = tf.convert_to_tensor([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]], tf.int32)
        self(input_ids, attention_mask, speaker_ids, duration_gts, f0_gts, energy_gts)

    def setup_f0_stat(self, min_f0=0.0, max_f0=7600.0, max_f0_embeddings=256, log_scale=True):
        """Setup min/max F0 for quantize."""
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.max_f0_embeddings = max_f0_embeddings

        if log_scale is True:
            self.min_f0 = np.log(self.min_f0 + 1e-5)
            self.max_f0 = np.log(self.max_f0 + 1e-5)

    def setup_energy_stat(self, min_energy, max_energy, max_energy_embeddings=256, log_scale=True):
        """Setup min/max energy for quantize."""
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.max_energy_embeddings = max_energy_embeddings

        if log_scale is True:
            self.min_energy = np.log(self.min_energy + 1e-5)
            self.max_energy = np.log(self.max_energy + 1e-5)

    def call(self,
             input_ids,
             attention_mask,
             speaker_ids,
             duration_gts,
             f0_gts,
             energy_gts,
             training=False):
        """Call logic."""
        embedding_output = self.embeddings([input_ids, speaker_ids], training=training)
        encoder_output = self.encoder([embedding_output, attention_mask], training=training)
        last_encoder_hidden_states = encoder_output[0]

        # energy predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for energy_predictor.
        duration_outputs = self.duration_predictor([last_encoder_hidden_states, attention_mask])  # [batch_size, length]

        length_regulator_outputs, encoder_masks = self.length_regulator([
            last_encoder_hidden_states, duration_gts], training=training)

        f0_outputs = self.f0_predictor(
            [length_regulator_outputs, encoder_masks], training=training)

        energy_outputs = self.energy_predictor(
            [length_regulator_outputs, encoder_masks], training=training)

        f0_embedding = self.f0_embeddings(f0_gts)  # [barch_size, mel_length, feature]
        energy_embedding = self.energy_embeddings(energy_gts)  # [barch_size, mel_length, feature]

        # sum features
        length_regulator_outputs = length_regulator_outputs + f0_embedding + energy_embedding

        # create decoder positional embedding
        decoder_pos = tf.range(1, tf.shape(length_regulator_outputs)[1] + 1, dtype=tf.int32)
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * encoder_masks

        decoder_output = self.decoder(
            [length_regulator_outputs, speaker_ids, encoder_masks, masked_decoder_pos], training=training)
        last_decoder_hidden_states = decoder_output[0]

        # here u can use sum or concat more than 1 hidden states layers from decoder.
        mel_before = self.mel_dense(last_decoder_hidden_states)
        mel_after = self.postnet([mel_before, encoder_masks], training=training) + mel_before

        outputs = (mel_before, mel_after, duration_outputs, f0_outputs, energy_outputs)
        return outputs

    @tf.function(experimental_relax_shapes=True,
                 input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None, None], dtype=tf.bool),
                                  tf.TensorSpec(shape=[None, ], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None, ], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, ], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, ], dtype=tf.float32)])
    def inference(self,
                  input_ids,
                  attention_mask,
                  speaker_ids,
                  speed_ratios,
                  f0_ratios,
                  energy_ratios):
        """Call logic."""
        embedding_output = self.embeddings([input_ids, speaker_ids], training=False)
        encoder_output = self.encoder([embedding_output, attention_mask], training=False)
        last_encoder_hidden_states = encoder_output[0]

        # energy predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for energy_predictor.
        duration_outputs = self.duration_predictor([last_encoder_hidden_states, attention_mask])  # [batch_size, length]
        duration_outputs = tf.math.exp(duration_outputs) - 1.0

        duration_outputs = tf.cast(tf.math.round(duration_outputs * speed_ratios), tf.int32)

        length_regulator_outputs, encoder_masks = self.length_regulator([
            last_encoder_hidden_states, duration_outputs], training=False)

        f0_outputs = self.f0_predictor(
            [length_regulator_outputs, encoder_masks], training=False)

        energy_outputs = self.energy_predictor(
            [length_regulator_outputs, encoder_masks], training=False)

        # scale f0/energy
        f0_outputs *= f0_ratios
        energy_outputs *= energy_ratios

        # quantize f0 and energy to category
        quantize_f0 = math_ops._bucketize(f0_outputs,
                                          boundaries=list(np.linspace(self.min_f0,
                                                                      self.max_f0,
                                                                      self.max_f0_embeddings)))
        quantize_energy = math_ops._bucketize(energy_outputs,
                                              boundaries=list(np.linspace(self.min_energy,
                                                                          self.max_energy,
                                                                          self.max_energy_embeddings)))

        f0_embedding = self.f0_embeddings(quantize_f0)  # [barch_size, mel_length, feature]
        energy_embedding = self.energy_embeddings(quantize_energy)  # [barch_size, mel_length, feature]

        # sum features
        length_regulator_outputs = length_regulator_outputs + f0_embedding + energy_embedding

        # create decoder positional embedding
        decoder_pos = tf.range(1, tf.shape(length_regulator_outputs)[1] + 1, dtype=tf.int32)
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * encoder_masks

        decoder_output = self.decoder(
            [length_regulator_outputs, speaker_ids, encoder_masks, masked_decoder_pos], training=False)
        last_decoder_hidden_states = decoder_output[0]

        # here u can use sum or concat more than 1 hidden states layers from decoder.
        mel_before = self.mel_dense(last_decoder_hidden_states)
        mel_after = self.postnet([mel_before, encoder_masks], training=False) + mel_before

        outputs = (mel_before, mel_after, duration_outputs, f0_outputs, energy_outputs)
        return outputs
