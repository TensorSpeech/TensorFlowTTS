# -*- coding: utf-8 -*-

# Copyright 2020 MINH ANH (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

"""Tensorflow Layer modules for Tacotron-2."""

import numpy as np
import collections

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


class TFTacotronEmbeddings(tf.keras.layers.Layer):
    """Construct character/phoneme/positional/speaker embeddings."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.embedding_hidden_size = config.embedding_hidden_size
        self.initializer_range = config.initializer_range

        self.speaker_embeddings = tf.keras.layers.Embedding(
            config.n_speakers,
            config.embedding_hidden_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="speaker_embeddings"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.embedding_dropout_prob)

    def build(self, input_shape):
        """Build shared character/phoneme embedding layers."""
        with tf.name_scope("character_embeddings"):
            self.character_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.embedding_hidden_size],
                initializer=get_initializer(self.initializer_range),
            )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Get character embeddings of inputs.

        Args:
            1. character, Tensor (int32) shape [batch_size, length].
            2. speaker_id, Tensor (int32) shape [batch_size]
        Returns:
            Tensor (float32) shape [batch_size, length, embedding_size].

        """
        return self._embedding(inputs, training=training)

    def _embedding(self, inputs, training=False):
        """Applies embedding based on inputs tensor."""
        input_ids, speaker_ids = inputs

        # create embeddings
        inputs_embeds = tf.gather(self.character_embeddings, input_ids)
        speaker_embeddings = self.speaker_embeddings(speaker_ids)

        # extended speaker embeddings
        extended_speaker_embeddings = speaker_embeddings[:, tf.newaxis, :]

        # sum all embedding
        embeddings = inputs_embeds + extended_speaker_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings


class TFTacotronConvBatchNorm(tf.keras.layers.Layer):
    """Tacotron-2 Convolutional Batchnorm module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_batch_norm = []
        for i in range(config.n_conv_encoder):
            conv = tf.keras.layers.Conv1D(
                filters=config.encoder_conv_filters,
                kernel_size=config.encoder_conv_kernel_sizes,
                padding='same',
                name='conv_._{}'.format(i)
            )
            batch_norm = tf.keras.layers.BatchNormalization(name='batch_norm_._{}'.format(i))
            self.conv_batch_norm.append((conv, batch_norm))
        self.activation = ACT2FN[config.encoder_activation]

    def call(self, inputs, training=False):
        """Call logic."""
        outputs, mask = inputs
        for conv, bn in self.conv_batch_norm:
            outputs = conv(outputs)
            outputs = bn(outputs)
            outputs = self.activation(outputs)
        return outputs * mask


class TFTacotronEncoder(tf.keras.layers.Layer):
    """Tacotron-2 Encoder."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.embeddings = TFTacotronEmbeddings(config, name='embeddings')
        self.convbn = TFTacotronConvBatchNorm(config, name='conv_batch_norm')
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=config.encoder_lstm_units, return_sequences=True),
            name='bilstm'
        )

    def call(self, inputs, training=False):
        """Call logic."""
        input_ids, speaker_ids, input_mask = inputs

        # create embedding and mask them since we sum
        # speaker embedding to all character embedding.
        extended_input_mask = tf.expand_dims(input_mask, -1)
        input_embeddings = self.embeddings([input_ids, speaker_ids], training=training)
        mask_embeddings = input_embeddings * extended_input_mask

        # pass embeddings to convolution batch norm
        conv_outputs = self.convbn([mask_embeddings, extended_input_mask], training=training)

        # bi-lstm.
        outputs = self.bilstm(conv_outputs)
        masked_outputs = outputs * extended_input_mask

        return masked_outputs


class TFTacotronLocationSensitiveAttention(tf.keras.layers.Layer):
    """Tacotron-2 location-sensitive attention."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(self, **kwargs)
        self.query_layer = tf.keras.layers.Dense(units=config.attention_dim,
                                                 use_bias=False,
                                                 name='query_layer')
        self.memory_layer = tf.keras.layers.Dense(units=config.memory_units,
                                                  use_bias=False,
                                                  name='memory_layer')
        self.location_convolution = tf.keras.layers.Conv1D(
            filters=config.attention_filters,
            kernel_size=config.attention_kernel,
            padding='same',
            name='location_conv'
        )
        self.location_layer = tf.keras.layers.Dense(units=config.attention_dim,
                                                    use_bias=False,
                                                    name='location_layer')
        self.config = config

    def build(self, input_shape):
        """Build weight and bias for location sensitive attention calculation."""
        with tf.name_scope("location_sensitive_variables"):
            self.weight = self.add_weight(
                "weight",
                shape=[self.config.attention_dim, ],
                initializer=get_initializer(self.config.initializer_range),
            )
            self.bias = self.add_weight(
                "bias",
                shape=[self.config.attention_dim, ],
                initializer="zeros",
            )
        super().build(input_shape)
    
    def setup_memory(self, memory):
        self.values = self.memory_layer(memory)

    def call(self, inputs, training=False):
        """Call logic."""
        query, memory, prev_alignments, input_mask = inputs
        processed_query = self.query_layer(query)
        extended_preprocessed_query = tf.expand_dims(processed_query, 1)  # [batch_size, 1, attention_dim]
        values = self.memory_layer(memory)  # [batch_size, max_len, attention_dim]
        extended_alignments = tf.expand_dims(prev_alignments, axis=2)  # [batch_size, max_len, 1]
        f_alignments = self.location_convolution(extended_alignments)
        processed_location_features = self.location_layer(f_alignments)  # [batch_size, max_len, attention_dim]

        # calculate attention scores
        energy = self._location_sensitive_score(extended_preprocessed_query,
                                                processed_location_features,
                                                values)  # [batch_size, max_len]
        # masking energy
        mask_energy = (1.0 - tf.cast(input_mask, tf.float32)) * -10000.0
        energy = energy + mask_energy  # [batch_size, max_len]

        # calculate attention scores (aka alignments)
        alignments = tf.nn.softmax(energy, axis=-1)  # [batch_size, max_len]

        # compute attention vector (aka context vector)
        extended_alignments = tf.expand_dims(alignments, axis=2)
        attention = tf.reduce_sum(extended_alignments * memory, axis=1)  # [batch_size, attention_dim]

        # cumulative alignments
        cum_alignments = alignments + prev_alignments

        return attention, alignments, cum_alignments

    def _location_sensitive_score(self, w_query, w_fil, w_keys):
        """Calculate location sensitive score."""
        score = self.weight * tf.nn.tanh(w_keys + w_query + w_fil + self.bias)
        return tf.reduce_sum(score, axis=-1)

    def get_initial_alignments(self, batch_size, max_time):
        """Get initial alignments."""
        return tf.zeros(shape=[batch_size, max_time], dtype=tf.float32)

    def get_initial_attention(self, batch_size):
        """Get initial attention."""
        return tf.zeros(shape=[batch_size, self.config.encoder_lstm_units * 2], dtype=tf.float32)


class TFTacotronPrenet(tf.keras.layers.Layer):
    """Tacotron-2 prenet."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.prenet_dense = [
            tf.keras.layers.Dense(units=config.prenet_units,
                                  activation=ACT2FN[config.prenet_activation],
                                  name='dense_._{}'.format(i))
            for i in range(config.n_prenet_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate=config.prenet_dropout_rate, name='dropout')

    def call(self, inputs, training=False):
        """Call logic."""
        outputs = inputs
        for layer in self.prenet_dense:
            outputs = layer(outputs)
            outputs = self.dropout(outputs, training=training)
        return outputs


class TFTacotronPostnet(tf.keras.layers.Layer):
    """Tacotron-2 prenet."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_batch_norm = []
        for i in range(config.n_conv_postnet):
            conv = tf.keras.layers.Conv1D(
                filters=config.postnet_conv_filters,
                kernel_size=config.postnet_conv_kernel_sizes,
                padding='same',
                name='conv_._{}'.format(i)
            )
            batch_norm = tf.keras.layers.BatchNormalization(name='batch_norm_._{}'.format(i))
            self.conv_batch_norm.append((conv, batch_norm))
        self.dropout = tf.keras.layers.Dropout(rate=config.postnet_dropout_rate, name='dropout')
        self.activation = [tf.nn.tanh] * (config.n_conv_postnet - 1) + [tf.identity]

    def call(self, inputs, training=False):
        """Call logic."""
        outputs, mask = inputs
        extended_mask = tf.expand_dims(mask, axis=2)
        for i, (conv, bn) in enumerate(self.conv_batch_norm):
            outputs = conv(outputs)
            outputs = bn(outputs)
            outputs = self.activation[i](outputs)
        return outputs * extended_mask


TFTacotronDecoderCellState = collections.namedtuple(
    'TFTacotronDecoderCellState',
    ['cell_state', 'attention', 'time', 'alignments', 'alignment_history'])


TFTacotronDecoderInput = collections.namedtuple(
    'TFTacotronDecoderInput', ['decoder_input', 'encoder_output', 'memory_lengths'])


class TFTacotronDecoderCell(tf.keras.layers.AbstractRNNCell):
    """Tacotron-2 custom decoder cell."""

    def __init__(self, config, training, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.training = training
        self.prenet = TFTacotronPrenet(config, name='prenet')

        # define lstm cell on decoder.
        # TODO(@dathudeptrai) switch to zone-out lstm.
        lstm_cells = []
        for i in range(config.n_lstm_decoder):
            lstm_cell = tf.keras.layers.LSTMCell(units=config.decoder_lstm_units,
                                                 name='lstm_cell_._{}'.format(i))
            lstm_cells.append(lstm_cell)
        self.stacked_lstm = tf.keras.layers.StackedRNNCells(lstm_cells,
                                                            name='stacked_lstm')

        # attention layer.
        self.attention_layer = TFTacotronLocationSensitiveAttention(config, name='location_sensitive_attention')

        # frame, stop projection layer.
        self.frame_projection = tf.keras.layers.Dense(units=config.n_mels, name='frame_projection')
        self.stop_projection = tf.keras.layers.Dense(units=1, name='stop_projection')

        self.config = config

    def build(self, input_shape):
        """Build."""
        self.batch_size = input_shape.decoder_input[0]
        self.alignment_size = input_shape.encoder_output[1]

    @property
    def output_size(self):
        """Return output (mel and stop_token) size."""
        return [self.config.n_mels, ], [1, ]

    @property
    def state_size(self):
        """Return hidden state size."""
        return TFTacotronDecoderCellState(
            cell_state=self.stacked_lstm.state_size,
            time=tf.TensorShape([]),
            attention=self.config.attention_dim,
            alignments=self.alignment_size,
            alignment_history=(),
        )

    def get_initial_state(self, batch_size, alignment_size):
        """Get initial states."""
        initial_lstm_cell_statess = self.stacked_lstm.get_initial_state(None, batch_size, dtype=tf.float32)
        initial_attention = self.attention_layer.get_initial_attention(batch_size)
        initial_alignments = self.attention_layer.get_initial_alignments(
            batch_size, alignment_size)
        initial_alignment_history = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        return TFTacotronDecoderCellState(
            cell_state=initial_lstm_cell_statess,
            time=tf.zeros([], dtype=tf.int32),
            attention=initial_attention,
            alignments=initial_alignments,
            alignment_history=initial_alignment_history
        )

    def call(self, inputs, states):
        """Call logic."""
        decoder_input, encoder_output, encoder_mask = tf.nest.flatten(inputs)

        # 1. apply prenet for decoder_input.
        prenet_out = self.prenet(decoder_input, training=self.training)  # [batch_size, dim]

        # 2. concat prenet_out and prev attention vector
        # then use it as input of stacked lstm layer.
        stacked_lstm_input = tf.concat([prenet_out, states.attention], axis=-1)
        stacked_lstm_output, next_cell_state = self.stacked_lstm(stacked_lstm_input, states.cell_state)

        # 3. compute attention, alignment and cumulative alignment.
        prev_alignments = states.alignments
        prev_alignment_history = states.alignment_history
        attention, alignments, cum_alignments = self.attention_layer(
            [stacked_lstm_output,
             encoder_output,
             prev_alignments,
             encoder_mask],
            training=self.training
        )

        # 4. compute frame feature and stop token.
        projection_inputs = tf.concat([stacked_lstm_output, attention], axis=-1)
        decoder_outputs = self.frame_projection(projection_inputs)
        stop_tokens = self.stop_projection(projection_inputs)

        # 5. save alignment history to visualize.
        alignment_history = prev_alignment_history.write(states.time, alignments)

        # 6. return new states.
        new_states = TFTacotronDecoderCellState(
            cell_state=next_cell_state,
            time=states.time + 1,
            attention=attention,
            alignments=cum_alignments,
            alignment_history=alignment_history
        )

        return (decoder_outputs, stop_tokens), new_states
