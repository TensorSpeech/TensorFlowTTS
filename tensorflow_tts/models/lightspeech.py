# -*- coding: utf-8 -*-
# Copyright 2020 The LightSpeech2 Authors and Minh Nguyen (@dathudeptrai)
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
"""Tensorflow Model modules for LightSpeech2."""

import tensorflow as tf
import numpy as np
from tensorflow_tts.models import BaseModel
from tensorflow_tts.models.fastspeech import (
    TFFastSpeechAttention,
    TFFastSpeechOutput,
    TFFastSpeechEmbeddings,
    TFFastSpeechLengthRegulator,
    ACT2FN,
    get_initializer,
    TFEmbedding,
)


class TFLightSpeechVariantPredictor(tf.keras.layers.Layer):
    """LightSpeech duration predictor module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_layers = []
        for i in range(config.variant_prediction_num_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.SeparableConv1D(
                    config.variant_predictor_filter,
                    config.variant_predictor_kernel_size,
                    padding="same",
                    name="conv_._{}".format(i),
                )
            )
            self.conv_layers.append(tf.keras.layers.Activation(tf.nn.relu))
            self.conv_layers.append(
                tf.keras.layers.LayerNormalization(
                    epsilon=config.layer_norm_eps, name="LayerNorm_._{}".format(i)
                )
            )
            self.conv_layers.append(
                tf.keras.layers.Dropout(config.variant_predictor_dropout_rate)
            )
        self.conv_layers_sequence = tf.keras.Sequential(self.conv_layers)
        self.output_layer = tf.keras.layers.Dense(1)

        if config.n_speakers > 1:
            self.decoder_speaker_embeddings = tf.keras.layers.Embedding(
                config.n_speakers,
                config.encoder_self_attention_params.hidden_size,
                embeddings_initializer=get_initializer(config.initializer_range),
                name="speaker_embeddings",
            )
            self.speaker_fc = tf.keras.layers.Dense(
                units=config.encoder_self_attention_params.hidden_size,
                name="speaker_fc",
            )

        self.config = config

    def call(self, inputs, training=False):
        """Call logic."""
        encoder_hidden_states, speaker_ids, attention_mask = inputs
        attention_mask = tf.cast(
            tf.expand_dims(attention_mask, 2), encoder_hidden_states.dtype
        )

        if self.config.n_speakers > 1:
            speaker_embeddings = self.decoder_speaker_embeddings(speaker_ids)
            speaker_features = tf.math.softplus(self.speaker_fc(speaker_embeddings))
            # extended speaker embeddings
            extended_speaker_features = speaker_features[:, tf.newaxis, :]
            encoder_hidden_states += extended_speaker_features

        # mask encoder hidden states
        masked_encoder_hidden_states = encoder_hidden_states * attention_mask

        # pass though first layer
        outputs = self.conv_layers_sequence(masked_encoder_hidden_states)
        outputs = self.output_layer(outputs)
        masked_outputs = outputs * attention_mask

        outputs = tf.squeeze(masked_outputs, -1)
        return outputs


class TFLightSpeechIntermediate(tf.keras.layers.Layer):
    """Intermediate representation module."""

    def __init__(self, config, index, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv1d_1 = tf.keras.layers.SeparableConv1D(
            config.intermediate_size,
            kernel_size=config.intermediate_kernel_size[index],
            kernel_initializer=get_initializer(config.initializer_range),
            padding="same",
            name="conv1d_1",
        )
        self.conv1d_2 = tf.keras.layers.SeparableConv1D(
            config.hidden_size,
            kernel_size=config.intermediate_kernel_size[index],
            kernel_initializer=get_initializer(config.initializer_range),
            padding="same",
            name="conv1d_2",
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, inputs):
        """Call logic."""
        hidden_states, attention_mask = inputs

        hidden_states = self.conv1d_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.conv1d_2(hidden_states)

        masked_hidden_states = hidden_states * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=hidden_states.dtype
        )
        return masked_hidden_states


class TFLightSpeechLayer(tf.keras.layers.Layer):
    """LightSpeech module (FFT module on the paper)."""

    def __init__(self, config, index, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.attention = TFFastSpeechAttention(config, name="attention")
        self.intermediate = TFLightSpeechIntermediate(
            config, index, name="intermediate"
        )
        self.bert_output = TFFastSpeechOutput(config, name="output")

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, attention_mask = inputs

        attention_outputs = self.attention(
            [hidden_states, attention_mask], training=training
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(
            [attention_output, attention_mask], training=training
        )
        layer_output = self.bert_output(
            [intermediate_output, attention_output], training=training
        )
        masked_layer_output = layer_output * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=layer_output.dtype
        )
        outputs = (masked_layer_output,) + attention_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class TFLightSpeechEncoder(tf.keras.layers.Layer):
    """Fast Speech encoder module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = [
            TFLightSpeechLayer(config, i, name="layer_._{}".format(i))
            for i in range(config.num_hidden_layers)
        ]

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, attention_mask = inputs

        all_hidden_states = ()
        all_attentions = ()
        for _, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                [hidden_states, attention_mask], training=training
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


class TFLightSpeechDecoder(TFLightSpeechEncoder):
    """Fast Speech decoder module."""

    def __init__(self, config, **kwargs):
        self.is_compatible_encoder = kwargs.pop("is_compatible_encoder", True)

        super().__init__(config, **kwargs)
        self.config = config

        # create decoder positional embedding
        self.decoder_positional_embeddings = TFEmbedding(
            config.max_position_embeddings + 1,
            config.hidden_size,
            weights=[self._sincos_embedding()],
            name="position_embeddings",
            trainable=False,
        )

        if self.is_compatible_encoder is False:
            self.project_compatible_decoder = tf.keras.layers.Dense(
                units=config.hidden_size, name="project_compatible_decoder"
            )

        if config.n_speakers > 1:
            self.decoder_speaker_embeddings = TFEmbedding(
                config.n_speakers,
                config.hidden_size,
                embeddings_initializer=get_initializer(config.initializer_range),
                name="speaker_embeddings",
            )
            self.speaker_fc = tf.keras.layers.Dense(
                units=config.hidden_size, name="speaker_fc"
            )

    def call(self, inputs, training=False):
        hidden_states, speaker_ids, encoder_mask, decoder_pos = inputs

        if self.is_compatible_encoder is False:
            hidden_states = self.project_compatible_decoder(hidden_states)

        # calculate new hidden states.
        hidden_states += tf.cast(
            self.decoder_positional_embeddings(decoder_pos), hidden_states.dtype
        )

        if self.config.n_speakers > 1:
            speaker_embeddings = self.decoder_speaker_embeddings(speaker_ids)
            speaker_features = tf.math.softplus(self.speaker_fc(speaker_embeddings))
            # extended speaker embeddings
            extended_speaker_features = speaker_features[:, tf.newaxis, :]
            hidden_states += extended_speaker_features

        return super().call([hidden_states, encoder_mask], training=training)

    def _sincos_embedding(self):
        position_enc = np.array(
            [
                [
                    pos / np.power(10000, 2.0 * (i // 2) / self.config.hidden_size)
                    for i in range(self.config.hidden_size)
                ]
                for pos in range(self.config.max_position_embeddings + 1)
            ]
        )

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        # pad embedding.
        position_enc[0] = 0.0

        return position_enc


class TFLightSpeech(BaseModel):
    """TF LightSpeech module."""

    def __init__(self, config, **kwargs):
        """Init layers for LightSpeech."""
        self.enable_tflite_convertible = kwargs.pop("enable_tflite_convertible", False)
        super().__init__(**kwargs)
        self.embeddings = TFFastSpeechEmbeddings(config, name="embeddings")
        self.encoder = TFLightSpeechEncoder(
            config.encoder_self_attention_params, name="encoder"
        )
        self.length_regulator = TFFastSpeechLengthRegulator(
            config,
            enable_tflite_convertible=self.enable_tflite_convertible,
            name="length_regulator",
        )
        self.decoder = TFLightSpeechDecoder(
            config.decoder_self_attention_params,
            is_compatible_encoder=config.encoder_self_attention_params.hidden_size
            == config.decoder_self_attention_params.hidden_size,
            name="decoder",
        )
        self.mel_dense = tf.keras.layers.Dense(
            units=config.num_mels, dtype=tf.float32, name="mel_dense"
        )

        self.setup_inference_fn()

        self.f0_predictor = TFLightSpeechVariantPredictor(
            config, dtype=tf.float32, name="f0_predictor"
        )
        self.duration_predictor = TFLightSpeechVariantPredictor(
            config, dtype=tf.float32, name="duration_predictor"
        )

        # define f0_embeddings and energy_embeddings
        self.f0_embeddings = tf.keras.layers.Conv1D(
            filters=config.encoder_self_attention_params.hidden_size,
            kernel_size=9,
            padding="same",
            name="f0_embeddings",
        )
        self.f0_dropout = tf.keras.layers.Dropout(0.5)

    def _build(self):
        """Dummy input for building model."""
        # fake inputs
        input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        duration_gts = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
        f0_gts = tf.convert_to_tensor(
            [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]], tf.float32
        )
        energy_gts = tf.convert_to_tensor(
            [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]], tf.float32
        )
        self(
            input_ids=input_ids,
            speaker_ids=speaker_ids,
            duration_gts=duration_gts,
            f0_gts=f0_gts,
            energy_gts=energy_gts,
        )

    def call(
        self,
        input_ids,
        speaker_ids,
        duration_gts,
        f0_gts,
        energy_gts,
        training=False,
        **kwargs,
    ):
        """Call logic."""
        attention_mask = tf.math.not_equal(input_ids, 0)
        embedding_output = self.embeddings([input_ids, speaker_ids], training=training)
        encoder_output = self.encoder(
            [embedding_output, attention_mask], training=training
        )
        last_encoder_hidden_states = encoder_output[0]

        # energy predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for energy_predictor.
        duration_outputs = self.duration_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask]
        )  # [batch_size, length]

        f0_outputs = self.f0_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask], training=training
        )

        f0_embedding = self.f0_embeddings(
            tf.expand_dims(f0_gts, 2)
        )  # [barch_size, mel_length, feature]
        # apply dropout both training/inference
        f0_embedding = self.f0_dropout(f0_embedding, training=True)

        # sum features
        last_encoder_hidden_states += f0_embedding

        length_regulator_outputs, encoder_masks = self.length_regulator(
            [last_encoder_hidden_states, duration_gts], training=training
        )

        # create decoder positional embedding
        decoder_pos = tf.range(
            1, tf.shape(length_regulator_outputs)[1] + 1, dtype=tf.int32
        )
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * encoder_masks

        decoder_output = self.decoder(
            [length_regulator_outputs, speaker_ids, encoder_masks, masked_decoder_pos],
            training=training,
        )
        last_decoder_hidden_states = decoder_output[0]

        # here u can use sum or concat more than 1 hidden states layers from decoder.
        mel_outputs = self.mel_dense(last_decoder_hidden_states)

        outputs = (mel_outputs, duration_outputs, f0_outputs)
        return outputs

    def _inference(
        self,
        input_ids,
        speaker_ids,
        speed_ratios,
        f0_ratios,
        energy_ratios,
        **kwargs,
    ):
        """Call logic."""
        attention_mask = tf.math.not_equal(input_ids, 0)
        embedding_output = self.embeddings([input_ids, speaker_ids], training=False)
        encoder_output = self.encoder(
            [embedding_output, attention_mask], training=False
        )
        last_encoder_hidden_states = encoder_output[0]

        # expand ratios
        speed_ratios = tf.expand_dims(speed_ratios, 1)  # [B, 1]
        f0_ratios = tf.expand_dims(f0_ratios, 1)  # [B, 1]
        energy_ratios = tf.expand_dims(energy_ratios, 1)  # [B, 1]

        # energy predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for energy_predictor.
        duration_outputs = self.duration_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask]
        )  # [batch_size, length]
        duration_outputs = tf.nn.relu(tf.math.exp(duration_outputs) - 1.0)
        duration_outputs = tf.cast(
            tf.math.round(duration_outputs * speed_ratios), tf.int32
        )

        f0_outputs = self.f0_predictor(
            [last_encoder_hidden_states, speaker_ids, attention_mask], training=False
        )
        f0_outputs *= f0_ratios
        f0_embedding = self.f0_embeddings(tf.expand_dims(f0_outputs, 2))

        # sum features
        last_encoder_hidden_states += f0_embedding

        length_regulator_outputs, encoder_masks = self.length_regulator(
            [last_encoder_hidden_states, duration_outputs], training=False
        )

        # create decoder positional embedding
        decoder_pos = tf.range(
            1, tf.shape(length_regulator_outputs)[1] + 1, dtype=tf.int32
        )
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * encoder_masks

        decoder_output = self.decoder(
            [length_regulator_outputs, speaker_ids, encoder_masks, masked_decoder_pos],
            training=False,
        )
        last_decoder_hidden_states = decoder_output[0]

        # here u can use sum or concat more than 1 hidden states layers from decoder.
        mel_outputs = self.mel_dense(last_decoder_hidden_states)

        outputs = (mel_outputs, duration_outputs, f0_outputs)
        return outputs

    def setup_inference_fn(self):
        self.inference = tf.function(
            self._inference,
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="input_ids"),
                tf.TensorSpec(
                    shape=[
                        None,
                    ],
                    dtype=tf.int32,
                    name="speaker_ids",
                ),
                tf.TensorSpec(
                    shape=[
                        None,
                    ],
                    dtype=tf.float32,
                    name="speed_ratios",
                ),
                tf.TensorSpec(
                    shape=[
                        None,
                    ],
                    dtype=tf.float32,
                    name="f0_ratios",
                ),
                tf.TensorSpec(
                    shape=[
                        None,
                    ],
                    dtype=tf.float32,
                    name="energy_ratios",
                ),
            ],
        )

        self.inference_tflite = tf.function(
            self._inference,
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape=[1, None], dtype=tf.int32, name="input_ids"),
                tf.TensorSpec(
                    shape=[
                        1,
                    ],
                    dtype=tf.int32,
                    name="speaker_ids",
                ),
                tf.TensorSpec(
                    shape=[
                        1,
                    ],
                    dtype=tf.float32,
                    name="speed_ratios",
                ),
                tf.TensorSpec(
                    shape=[
                        1,
                    ],
                    dtype=tf.float32,
                    name="f0_ratios",
                ),
                tf.TensorSpec(
                    shape=[
                        1,
                    ],
                    dtype=tf.float32,
                    name="energy_ratios",
                ),
            ],
        )
