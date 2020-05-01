# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen Quan Anh (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)
"""Tensorflow Layer modules for FastSpeech."""

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
    return x * tf.sigmoid(x)


ACT2FN = {
    "gelu": tf.keras.layers.Activation(gelu),
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.layers.Activation(swish),
    "gelu_new": tf.keras.layers.Activation(gelu_new),
}


class TFFastSpeechEmbeddings(tf.keras.layers.Layer):
    """Construct charactor/phoneme/positional/speaker embeddings."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.config = config

        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings + 1,
            config.hidden_size,
            weights=[self._sincos_embedding()],
            name="position_embeddings",
            trainable=False,
        )

        if config.n_speakers > 1:
            self.speaker_embeddings = tf.keras.layers.Embedding(
                config.n_speakers,
                config.hidden_size,
                embeddings_initializer=get_initializer(self.initializer_range),
                name="speaker_embeddings"
            )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

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
        embeddings = inputs_embeds + position_embeddings
        if self.config.n_speakers > 1:
            speaker_embeddings = self.speaker_embeddings(speaker_ids)
            # extended speaker embeddings
            extended_speaker_embeddings = speaker_embeddings[:, tf.newaxis, :]
            embeddings += extended_speaker_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def _sincos_embedding(self):
        position_enc = np.array([
            [pos / np.power(10000, 2.0 * i / self.hidden_size) for i in range(self.hidden_size)]
            for pos in range(self.config.max_position_embeddings + 1)
        ])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        # pad embedding.
        position_enc[0] = 0.0

        return position_enc


class TFTacotronConvBatchNorm(tf.keras.layers.Layer):
    """Tacotron-2 Convolutional Batchnorm module."""
    def __init__(self, epsilon, filters, kernel_size, dropout_rate, activation=None, name_idx=None):
        super(TFTacotronConvBatchNorm, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(filters, kernel_size, padding='same', name='conv_._'.format(name_idx))
        self.norm = tf.keras.layers.LayerNormalization(epsilon=epsilon, name='LayerNorm._{}'.format(name_idx))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout_._'.format(name_idx))
        self.act = ACT2FN[activation]

    def call(self, x, training=False):
        o = self.conv1d(x)
        o = self.norm(o)
        o = self.act(o)
        o = self.dropout(o, training=training)
        return o

class TFTacotronEncoderConvs(tf.keras.layers.Layer):
    """Tacotron-2 Encoder Convolutional Batchnorm module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_batch_norm = []
        for i in range(config.n_conv_encoder):
            conv = TFTacotronConvBatchNorm(
                epsilon=config.layer_norm_eps, 
                filters=config.encoder_conv_filters,
                kernel_size=config.encoder_conv_kernel_sizes,
                activation=config.encoder_conv_activation,
                dropout_rate=config.encoder_conv_dropout_rate,
                name_idx=i)
            self.conv_batch_norm.append(conv)

    def call(self, inputs, training=False):
        """Call logic."""
        outputs, mask = inputs
        for conv in self.conv_batch_norm:
            outputs = conv(outputs, training=training)
        return outputs * tf.cast(tf.expand_dims(mask, -1), tf.float32)


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
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, batch_size):
        """Transpose to calculate attention scores."""
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
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
        dk = tf.cast(tf.shape(key_layer)[-1], tf.float32)  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # extended_attention_masks for self attention encoder.
            extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
            extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
            attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer, (batch_size, -1, self.all_head_size)
        )

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class TFFastSpeechSelfOutput(tf.keras.layers.Layer):
    """Fastspeech output of self attention module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
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

        self_outputs = self.self_attention([input_tensor, attention_mask], training=training)
        attention_output = self.dense_output([self_outputs[0], input_tensor], training=training)
        masked_attention_output = attention_output * tf.cast(tf.expand_dims(attention_mask, 2), dtype=tf.float32)
        outputs = (masked_attention_output,) + self_outputs[1:]  # add attentions if we output them
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
            padding='same',
            name="conv1d_1"
        )
        self.conv1d_2 = tf.keras.layers.Conv1D(
            config.hidden_size,
            kernel_size=config.intermediate_kernel_size,
            kernel_initializer=get_initializer(config.initializer_range),
            padding='same',
            name="conv1d_2"
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

        masked_hidden_states = hidden_states * tf.cast(tf.expand_dims(attention_mask, 2), dtype=tf.float32)
        return masked_hidden_states


class TFFastSpeechOutput(tf.keras.layers.Layer):
    """Output module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
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

        attention_outputs = self.attention([hidden_states, attention_mask], training=training)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate([attention_output, attention_mask], training=training)
        layer_output = self.bert_output([intermediate_output, attention_output], training=training)
        masked_layer_output = layer_output * tf.cast(tf.expand_dims(attention_mask, 2), dtype=tf.float32)
        outputs = (masked_layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class TFFastSpeechEncoder(tf.keras.layers.Layer):
    """Fast Speech encoder module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = [TFFastSpeechLayer(config, name="layer_._{}".format(i)) for i in range(config.num_hidden_layers)]

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, attention_mask = inputs

        all_hidden_states = ()
        all_attentions = ()
        for _, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module([hidden_states, attention_mask], training=training)
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
        super().__init__(config, **kwargs)
        self.config = config

        # create decoder positional embedding
        self.decoder_positional_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings + 1,
            config.hidden_size,
            weights=[self._sincos_embedding()],
            name="position_embeddings",
            trainable=False
        )

    def call(self, inputs, training=False):
        hidden_states, decoder_mask, decoder_pos = inputs

        # calculate new hidden states.
        hidden_states = hidden_states + self.decoder_positional_embeddings(decoder_pos)
        return super().call([hidden_states, decoder_mask], training=training)

    def _sincos_embedding(self):
        position_enc = np.array([
            [pos / np.power(10000, 2.0 * i / self.config.hidden_size) for i in range(self.config.hidden_size)]
            for pos in range(self.config.max_position_embeddings + 1)
        ])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        # pad embedding.
        position_enc[0] = 0.0

        return position_enc


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
                    padding='same',
                    name='conv_._{}'.format(i)
                )
            )
            self.conv_layers.append(
                tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm_._{}".format(i))
            )
            self.conv_layers.append(
                tf.keras.layers.Activation(tf.nn.relu6)
            )
            self.conv_layers.append(
                tf.keras.layers.Dropout(config.duration_predictor_dropout_probs)
            )
        self.conv_layers_sequence = tf.keras.Sequential(self.conv_layers)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        """Call logic."""
        encoder_hidden_states, attention_mask = inputs
        attention_mask = tf.cast(tf.expand_dims(attention_mask, 2), tf.float32)

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
        super().__init__(**kwargs)
        self.config = config

    def call(self, inputs, training=False):
        """Call logic.

        Args:
            1. encoder_hidden_states, Tensor (float32) shape [batch_size, length, hidden_size]
            2. durations_gt, Tensor (float32/int32) shape [batch_size, length]
        """
        encoder_hidden_states, durations_gt = inputs
        outputs, encoder_masks = self._length_regulator(encoder_hidden_states, durations_gt)
        return outputs, encoder_masks

    def _length_regulator(self, encoder_hidden_states, durations_gt):
        """Length regulator logic."""
        sum_durations = tf.reduce_sum(durations_gt, axis=-1)  # [batch_size]
        max_durations = tf.reduce_max(sum_durations)

        input_shape = tf.shape(encoder_hidden_states)
        batch_size = input_shape[0]
        hidden_size = input_shape[-1]

        # initialize output hidden states and encoder masking.
        outputs = tf.zeros(shape=[0, max_durations, hidden_size], dtype=tf.float32)
        encoder_masks = tf.zeros(shape=[0, max_durations], dtype=tf.int32)

        def condition(i,
                      batch_size,
                      outputs,
                      encoder_masks,
                      encoder_hidden_states,
                      durations_gt,
                      max_durations):
            return tf.less(i, batch_size)

        def body(i,
                 batch_size,
                 outputs,
                 encoder_masks,
                 encoder_hidden_states,
                 durations_gt,
                 max_durations):
            repeats = durations_gt[i]
            real_length = tf.reduce_sum(repeats)
            pad_size = max_durations - real_length
            masks = tf.sequence_mask([real_length], max_durations, dtype=tf.int32)
            repeat_encoder_hidden_states = tf.repeat(
                encoder_hidden_states[i],
                repeats=repeats,
                axis=0
            )
            repeat_encoder_hidden_states = tf.expand_dims(
                tf.pad(
                    repeat_encoder_hidden_states, [[0, pad_size], [0, 0]]
                ),
                0)  # [1, max_durations, hidden_size]
            outputs = tf.concat([outputs, repeat_encoder_hidden_states], axis=0)
            encoder_masks = tf.concat([encoder_masks, masks], axis=0)
            return [i + 1, batch_size, outputs, encoder_masks,
                    encoder_hidden_states, durations_gt, max_durations]

        # initialize iteration i.
        i = tf.constant(0, dtype=tf.int32)
        _, _, outputs, encoder_masks, _, _, _, = tf.while_loop(
            condition,
            body,
            [i, batch_size, outputs, encoder_masks, encoder_hidden_states, durations_gt, max_durations],
            shape_invariants=[i.get_shape(),
                              batch_size.get_shape(),
                              tf.TensorShape([None, None, self.config.hidden_size]),
                              tf.TensorShape([None, None]),
                              encoder_hidden_states.get_shape(),
                              durations_gt.get_shape(),
                              max_durations.get_shape()]
        )

        return outputs, encoder_masks
