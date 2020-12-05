# -*- coding: utf-8 -*-
# Copyright 2020 The FastSpeech Authors, The HuggingFace Inc. team and Minh Nguyen (@dathudeptrai)
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
"""Tensorflow Model modules for FastSpeech."""

import numpy as np
import tensorflow as tf


def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.

    Args:
        initializer_range: float, initializer range for stddev.

    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.

    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def gelu(x):
    """Gaussian Error Linear unit."""
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    return x * cdf


def gelu_new(x):
    """Smoother gaussian Error Linear Unit."""
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def swish(x):
    """Swish activation function."""
    return tf.nn.swish(x)


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


ACT2FN = {
    "identity": tf.keras.layers.Activation("linear"),
    "tanh": tf.keras.layers.Activation("tanh"),
    "gelu": tf.keras.layers.Activation(gelu),
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.layers.Activation(swish),
    "gelu_new": tf.keras.layers.Activation(gelu_new),
    "mish": tf.keras.layers.Activation(mish),
}


class TFEmbedding(tf.keras.layers.Embedding):
    """Faster version of embedding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.int32)
        outputs = tf.gather(self.embeddings, inputs)
        return outputs


class TFFastSpeechEmbeddings(tf.keras.layers.Layer):
    """Construct charactor/phoneme/positional/speaker embeddings."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.encoder_self_attention_params.hidden_size
        self.initializer_range = config.initializer_range
        self.config = config

        self.position_embeddings = TFEmbedding(
            config.max_position_embeddings + 1,
            self.hidden_size,
            weights=[
                self._sincos_embedding(
                    self.hidden_size, self.config.max_position_embeddings
                )
            ],
            name="position_embeddings",
            trainable=False,
        )

        if config.n_speakers > 1:
            self.encoder_speaker_embeddings = TFEmbedding(
                config.n_speakers,
                self.hidden_size,
                embeddings_initializer=get_initializer(self.initializer_range),
                name="speaker_embeddings",
            )
            self.speaker_fc = tf.keras.layers.Dense(
                units=self.hidden_size, name="speaker_fc"
            )

    def build(self, input_shape):
        """Build shared charactor/phoneme embedding layers."""
        with tf.name_scope("charactor_embeddings"):
            self.charactor_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Get charactor embeddings of inputs.

        Args:
            1. charactor, Tensor (int32) shape [batch_size, length].
            2. speaker_id, Tensor (int32) shape [batch_size]
        Returns:
            Tensor (float32) shape [batch_size, length, embedding_size].

        """
        return self._embedding(inputs, training=training)

    def _embedding(self, inputs, training=False):
        """Applies embedding based on inputs tensor."""
        input_ids, speaker_ids = inputs

        input_shape = tf.shape(input_ids)
        seq_length = input_shape[1]

        position_ids = tf.range(1, seq_length + 1, dtype=tf.int32)[tf.newaxis, :]

        # create embeddings
        inputs_embeds = tf.gather(self.charactor_embeddings, input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # sum embedding
        embeddings = inputs_embeds + tf.cast(position_embeddings, inputs_embeds.dtype)
        if self.config.n_speakers > 1:
            speaker_embeddings = self.encoder_speaker_embeddings(speaker_ids)
            speaker_features = tf.math.softplus(self.speaker_fc(speaker_embeddings))
            # extended speaker embeddings
            extended_speaker_features = speaker_features[:, tf.newaxis, :]
            embeddings += extended_speaker_features

        return embeddings

    def _sincos_embedding(
        self, hidden_size, max_positional_embedding,
    ):
        position_enc = np.array(
            [
                [
                    pos / np.power(10000, 2.0 * (i // 2) / hidden_size)
                    for i in range(hidden_size)
                ]
                for pos in range(max_positional_embedding + 1)
            ]
        )

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        # pad embedding.
        position_enc[0] = 0.0

        return position_enc

    def resize_positional_embeddings(self, new_size):
        self.position_embeddings = TFEmbedding(
            new_size + 1,
            self.hidden_size,
            weights=[self._sincos_embedding(self.hidden_size, new_size)],
            name="position_embeddings",
            trainable=False,
        )


class TFFastSpeechSelfAttention(tf.keras.layers.Layer):
    """Self attention module for fastspeech."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * config.attention_head_size

        self.query = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="query",
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="key",
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="value",
        )

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.config = config

    def transpose_for_scores(self, x, batch_size):
        """Transpose to calculate attention scores."""
        x = tf.reshape(
            x,
            (batch_size, -1, self.num_attention_heads, self.config.attention_head_size),
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, attention_mask = inputs

        batch_size = tf.shape(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(
            tf.shape(key_layer)[-1], attention_scores.dtype
        )  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # extended_attention_masks for self attention encoder.
            extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
            extended_attention_mask = tf.cast(
                extended_attention_mask, attention_scores.dtype
            )
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
            attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.all_head_size))

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class TFFastSpeechSelfOutput(tf.keras.layers.Layer):
    """Fastspeech output of self attention module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, input_tensor = inputs

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFFastSpeechAttention(tf.keras.layers.Layer):
    """Fastspeech attention module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.self_attention = TFFastSpeechSelfAttention(config, name="self")
        self.dense_output = TFFastSpeechSelfOutput(config, name="output")

    def call(self, inputs, training=False):
        input_tensor, attention_mask = inputs

        self_outputs = self.self_attention(
            [input_tensor, attention_mask], training=training
        )
        attention_output = self.dense_output(
            [self_outputs[0], input_tensor], training=training
        )
        masked_attention_output = attention_output * tf.cast(
            tf.expand_dims(attention_mask, 2), dtype=attention_output.dtype
        )
        outputs = (masked_attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class TFFastSpeechIntermediate(tf.keras.layers.Layer):
    """Intermediate representation module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv1d_1 = tf.keras.layers.Conv1D(
            config.intermediate_size,
            kernel_size=config.intermediate_kernel_size,
            kernel_initializer=get_initializer(config.initializer_range),
            padding="same",
            name="conv1d_1",
        )
        self.conv1d_2 = tf.keras.layers.Conv1D(
            config.hidden_size,
            kernel_size=config.intermediate_kernel_size,
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


class TFFastSpeechOutput(tf.keras.layers.Layer):
    """Output module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="LayerNorm"
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, input_tensor = inputs

        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TFFastSpeechLayer(tf.keras.layers.Layer):
    """Fastspeech module (FFT module on the paper)."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.attention = TFFastSpeechAttention(config, name="attention")
        self.intermediate = TFFastSpeechIntermediate(config, name="intermediate")
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


class TFFastSpeechEncoder(tf.keras.layers.Layer):
    """Fast Speech encoder module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = [
            TFFastSpeechLayer(config, name="layer_._{}".format(i))
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


class TFFastSpeechDecoder(TFFastSpeechEncoder):
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


class TFTacotronPostnet(tf.keras.layers.Layer):
    """Tacotron-2 postnet."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_batch_norm = []
        for i in range(config.n_conv_postnet):
            conv = tf.keras.layers.Conv1D(
                filters=config.postnet_conv_filters
                if i < config.n_conv_postnet - 1
                else config.num_mels,
                kernel_size=config.postnet_conv_kernel_sizes,
                padding="same",
                name="conv_._{}".format(i),
            )
            batch_norm = tf.keras.layers.BatchNormalization(
                axis=-1, name="batch_norm_._{}".format(i)
            )
            self.conv_batch_norm.append((conv, batch_norm))
        self.dropout = tf.keras.layers.Dropout(
            rate=config.postnet_dropout_rate, name="dropout"
        )
        self.activation = [tf.nn.tanh] * (config.n_conv_postnet - 1) + [tf.identity]

    def call(self, inputs, training=False):
        """Call logic."""
        outputs, mask = inputs
        extended_mask = tf.cast(tf.expand_dims(mask, axis=2), outputs.dtype)
        for i, (conv, bn) in enumerate(self.conv_batch_norm):
            outputs = conv(outputs)
            outputs = bn(outputs)
            outputs = self.activation[i](outputs)
            outputs = self.dropout(outputs, training=training)
        return outputs * extended_mask


class TFFastSpeechDurationPredictor(tf.keras.layers.Layer):
    """FastSpeech duration predictor module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_layers = []
        for i in range(config.num_duration_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv1D(
                    config.duration_predictor_filters,
                    config.duration_predictor_kernel_sizes,
                    padding="same",
                    name="conv_._{}".format(i),
                )
            )
            self.conv_layers.append(
                tf.keras.layers.LayerNormalization(
                    epsilon=config.layer_norm_eps, name="LayerNorm_._{}".format(i)
                )
            )
            self.conv_layers.append(tf.keras.layers.Activation(tf.nn.relu6))
            self.conv_layers.append(
                tf.keras.layers.Dropout(config.duration_predictor_dropout_probs)
            )
        self.conv_layers_sequence = tf.keras.Sequential(self.conv_layers)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        """Call logic."""
        encoder_hidden_states, attention_mask = inputs
        attention_mask = tf.cast(
            tf.expand_dims(attention_mask, 2), encoder_hidden_states.dtype
        )

        # mask encoder hidden states
        masked_encoder_hidden_states = encoder_hidden_states * attention_mask

        # pass though first layer
        outputs = self.conv_layers_sequence(masked_encoder_hidden_states)
        outputs = self.output_layer(outputs)
        masked_outputs = outputs * attention_mask
        return tf.squeeze(tf.nn.relu6(masked_outputs), -1)  # make sure positive value.


class TFFastSpeechLengthRegulator(tf.keras.layers.Layer):
    """FastSpeech lengthregulator module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        self.enable_tflite_convertible = kwargs.pop("enable_tflite_convertible", False)
        super().__init__(**kwargs)
        self.config = config

    def call(self, inputs, training=False):
        """Call logic.
        Args:
            1. encoder_hidden_states, Tensor (float32) shape [batch_size, length, hidden_size]
            2. durations_gt, Tensor (float32/int32) shape [batch_size, length]
        """
        encoder_hidden_states, durations_gt = inputs
        outputs, encoder_masks = self._length_regulator(
            encoder_hidden_states, durations_gt
        )
        return outputs, encoder_masks

    def _length_regulator(self, encoder_hidden_states, durations_gt):
        """Length regulator logic."""
        sum_durations = tf.reduce_sum(durations_gt, axis=-1)  # [batch_size]
        max_durations = tf.reduce_max(sum_durations)

        input_shape = tf.shape(encoder_hidden_states)
        batch_size = input_shape[0]
        hidden_size = input_shape[-1]

        # initialize output hidden states and encoder masking.
        if self.enable_tflite_convertible:
            # There is only 1 batch in inference, so we don't have to use
            # `tf.While` op with 3-D output tensor.
            repeats = durations_gt[0]
            real_length = tf.reduce_sum(repeats)
            pad_size = max_durations - real_length
            # masks : [max_durations]
            masks = tf.sequence_mask([real_length], max_durations, dtype=tf.int32)
            repeat_encoder_hidden_states = tf.repeat(
                encoder_hidden_states[0], repeats=repeats, axis=0
            )
            repeat_encoder_hidden_states = tf.expand_dims(
                tf.pad(repeat_encoder_hidden_states, [[0, pad_size], [0, 0]]), 0
            )  # [1, max_durations, hidden_size]

            outputs = repeat_encoder_hidden_states
            encoder_masks = masks
        else:
            outputs = tf.zeros(
                shape=[0, max_durations, hidden_size], dtype=encoder_hidden_states.dtype
            )
            encoder_masks = tf.zeros(shape=[0, max_durations], dtype=tf.int32)

            def condition(
                i,
                batch_size,
                outputs,
                encoder_masks,
                encoder_hidden_states,
                durations_gt,
                max_durations,
            ):
                return tf.less(i, batch_size)

            def body(
                i,
                batch_size,
                outputs,
                encoder_masks,
                encoder_hidden_states,
                durations_gt,
                max_durations,
            ):
                repeats = durations_gt[i]
                real_length = tf.reduce_sum(repeats)
                pad_size = max_durations - real_length
                masks = tf.sequence_mask([real_length], max_durations, dtype=tf.int32)
                repeat_encoder_hidden_states = tf.repeat(
                    encoder_hidden_states[i], repeats=repeats, axis=0
                )
                repeat_encoder_hidden_states = tf.expand_dims(
                    tf.pad(repeat_encoder_hidden_states, [[0, pad_size], [0, 0]]), 0
                )  # [1, max_durations, hidden_size]
                outputs = tf.concat([outputs, repeat_encoder_hidden_states], axis=0)
                encoder_masks = tf.concat([encoder_masks, masks], axis=0)
                return [
                    i + 1,
                    batch_size,
                    outputs,
                    encoder_masks,
                    encoder_hidden_states,
                    durations_gt,
                    max_durations,
                ]

            # initialize iteration i.
            i = tf.constant(0, dtype=tf.int32)
            _, _, outputs, encoder_masks, _, _, _, = tf.while_loop(
                condition,
                body,
                [
                    i,
                    batch_size,
                    outputs,
                    encoder_masks,
                    encoder_hidden_states,
                    durations_gt,
                    max_durations,
                ],
                shape_invariants=[
                    i.get_shape(),
                    batch_size.get_shape(),
                    tf.TensorShape(
                        [
                            None,
                            None,
                            self.config.encoder_self_attention_params.hidden_size,
                        ]
                    ),
                    tf.TensorShape([None, None]),
                    encoder_hidden_states.get_shape(),
                    durations_gt.get_shape(),
                    max_durations.get_shape(),
                ],
            )

        return outputs, encoder_masks


class TFFastSpeech(tf.keras.Model):
    """TF Fastspeech module."""

    def __init__(self, config, **kwargs):
        """Init layers for fastspeech."""
        self.enable_tflite_convertible = kwargs.pop("enable_tflite_convertible", False)
        super().__init__(**kwargs)
        self.embeddings = TFFastSpeechEmbeddings(config, name="embeddings")
        self.encoder = TFFastSpeechEncoder(
            config.encoder_self_attention_params, name="encoder"
        )
        self.duration_predictor = TFFastSpeechDurationPredictor(
            config, dtype=tf.float32, name="duration_predictor"
        )
        self.length_regulator = TFFastSpeechLengthRegulator(
            config,
            enable_tflite_convertible=self.enable_tflite_convertible,
            name="length_regulator",
        )
        self.decoder = TFFastSpeechDecoder(
            config.decoder_self_attention_params,
            is_compatible_encoder=config.encoder_self_attention_params.hidden_size
            == config.decoder_self_attention_params.hidden_size,
            name="decoder",
        )
        self.mel_dense = tf.keras.layers.Dense(
            units=config.num_mels, dtype=tf.float32, name="mel_before"
        )
        self.postnet = TFTacotronPostnet(
            config=config, dtype=tf.float32, name="postnet"
        )

        self.setup_inference_fn()

    def _build(self):
        """Dummy input for building model."""
        # fake inputs
        input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        duration_gts = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
        self(input_ids, speaker_ids, duration_gts)

    def resize_positional_embeddings(self, new_size):
        self.embeddings.resize_positional_embeddings(new_size)
        self._build()

    def call(
        self, input_ids, speaker_ids, duration_gts, training=False, **kwargs,
    ):
        """Call logic."""
        attention_mask = tf.math.not_equal(input_ids, 0)
        embedding_output = self.embeddings([input_ids, speaker_ids], training=training)
        encoder_output = self.encoder(
            [embedding_output, attention_mask], training=training
        )
        last_encoder_hidden_states = encoder_output[0]

        # duration predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for duration_predictor.
        duration_outputs = self.duration_predictor(
            [last_encoder_hidden_states, attention_mask]
        )  # [batch_size, length]

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
        mel_before = self.mel_dense(last_decoder_hidden_states)
        mel_after = (
            self.postnet([mel_before, encoder_masks], training=training) + mel_before
        )

        outputs = (mel_before, mel_after, duration_outputs)
        return outputs

    def _inference(self, input_ids, speaker_ids, speed_ratios, **kwargs):
        """Call logic."""
        attention_mask = tf.math.not_equal(input_ids, 0)
        embedding_output = self.embeddings([input_ids, speaker_ids], training=False)
        encoder_output = self.encoder(
            [embedding_output, attention_mask], training=False
        )
        last_encoder_hidden_states = encoder_output[0]

        # duration predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for duration_predictor.
        duration_outputs = self.duration_predictor(
            [last_encoder_hidden_states, attention_mask]
        )  # [batch_size, length]
        duration_outputs = tf.math.exp(duration_outputs) - 1.0

        if speed_ratios is None:
            speed_ratios = tf.convert_to_tensor(np.array([1.0]), dtype=tf.float32)

        speed_ratios = tf.expand_dims(speed_ratios, 1)

        duration_outputs = tf.cast(
            tf.math.round(duration_outputs * speed_ratios), tf.int32
        )

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
        mel_before = self.mel_dense(last_decoder_hidden_states)
        mel_after = (
            self.postnet([mel_before, encoder_masks], training=False) + mel_before
        )

        outputs = (mel_before, mel_after, duration_outputs)
        return outputs

    def setup_inference_fn(self):
        self.inference = tf.function(
            self._inference,
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="input_ids"),
                tf.TensorSpec(shape=[None,], dtype=tf.int32, name="speaker_ids"),
                tf.TensorSpec(shape=[None,], dtype=tf.float32, name="speed_ratios"),
            ],
        )

        self.inference_tflite = tf.function(
            self._inference,
            experimental_relax_shapes=True,
            input_signature=[
                tf.TensorSpec(shape=[1, None], dtype=tf.int32, name="input_ids"),
                tf.TensorSpec(shape=[1,], dtype=tf.int32, name="speaker_ids"),
                tf.TensorSpec(shape=[1,], dtype=tf.float32, name="speed_ratios"),
            ],
        )
