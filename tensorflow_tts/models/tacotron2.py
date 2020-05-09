# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen (@dathudeptrai) - Eren Gölge (@erogol)
#  MIT License (https://opensource.org/licenses/MIT)

"""Tacotron-2 Modules."""

import collections
import numpy as np
from functools import partial

import tensorflow as tf

from tensorflow_addons.seq2seq import Sampler
from tensorflow_addons.seq2seq import BahdanauAttention
from tensorflow_addons.seq2seq import dynamic_decode
from tensorflow_addons.seq2seq import Decoder
from tensorflow_addons.seq2seq import AttentionMechanism
from tensorflow_addons.seq2seq import attention_wrapper


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


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


ACT2FN = {
    "identity": tf.keras.layers.Activation('linear'),
    "tanh": tf.keras.layers.Activation('tanh'),
    "gelu": tf.keras.layers.Activation(gelu),
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.layers.Activation(swish),
    "gelu_new": tf.keras.layers.Activation(gelu_new),
    "mish": tf.keras.layers.Activation(mish)
}


class TFTacotronConvBatchNorm(tf.keras.layers.Layer):
    """Tacotron-2 Convolutional Batchnorm module."""

    def __init__(self, filters, kernel_size, dropout_rate, activation=None, name_idx=None):
        super().__init__()
        self.conv1d = tf.keras.layers.Conv1D(filters,
                                             kernel_size,
                                             kernel_initializer=get_initializer(0.02),
                                             padding='same',
                                             name='conv_._{}'.format(name_idx))
        self.norm = tf.keras.layers.BatchNormalization(axis=-1, name='batch_norm_._{}'.format(name_idx))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout_._{}'.format(name_idx))
        self.act = ACT2FN[activation]

    def call(self, inputs, training=False):
        outputs = self.conv1d(inputs)
        outputs = self.norm(outputs, training=training)
        outputs = self.act(outputs)
        outputs = self.dropout(outputs, training=training)
        return outputs


class TFTacotronEmbeddings(tf.keras.layers.Layer):
    """Construct character/phoneme/positional/speaker embeddings."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.embedding_hidden_size = config.embedding_hidden_size
        self.initializer_range = config.initializer_range
        self.config = config

        if config.n_speakers > 1:
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
        embeddings = inputs_embeds

        if self.config.n_speakers > 1:
            speaker_embeddings = self.speaker_embeddings(speaker_ids)
            extended_speaker_embeddings = speaker_embeddings[:, tf.newaxis, :]
            # sum all embedding
            embeddings += extended_speaker_embeddings

        # apply layer-norm and dropout for embeddings.
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)

        return embeddings


class TFTacotronEncoderConvs(tf.keras.layers.Layer):
    """Tacotron-2 Encoder Convolutional Batchnorm module."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_batch_norm = []
        for i in range(config.n_conv_encoder):
            conv = TFTacotronConvBatchNorm(
                filters=config.encoder_conv_filters,
                kernel_size=config.encoder_conv_kernel_sizes,
                activation=config.encoder_conv_activation,
                dropout_rate=config.encoder_conv_dropout_rate,
                name_idx=i)
            self.conv_batch_norm.append(conv)

    def call(self, inputs, training=False):
        """Call logic."""
        outputs = inputs
        for conv in self.conv_batch_norm:
            outputs = conv(outputs, training=training)
        return outputs


class TFTacotronEncoder(tf.keras.layers.Layer):
    """Tacotron-2 Encoder."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.embeddings = TFTacotronEmbeddings(config, name='embeddings')
        self.convbn = TFTacotronEncoderConvs(config, name='conv_batch_norm')
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=config.encoder_lstm_units, return_sequences=True),
            name='bilstm'
        )

    def call(self, inputs, training=False):
        """Call logic."""
        input_ids, speaker_ids, input_mask = inputs

        # create embedding and mask them since we sum
        # speaker embedding to all character embedding.
        input_embeddings = self.embeddings([input_ids, speaker_ids], training=training)

        # pass embeddings to convolution batch norm
        conv_outputs = self.convbn(input_embeddings, training=training)

        # bi-lstm.
        outputs = self.bilstm(conv_outputs, mask=input_mask)

        return outputs


class TrainingSampler(Sampler):
    """Training sampler for Seq2Seq training."""

    def __init__(self,
                 config,
                 ):
        super().__init__()
        self.config = config
        # create schedule factor.
        # the input of a next decoder cell is calculated by formular:
        # next_inputs = ratio * prev_groundtruth_outputs + (1.0 - ratio) * prev_predicted_outputs.
        self._ratio = tf.constant(1.0)

    def setup_target(self, targets, mel_lengths):
        """Setup ground-truth mel outputs for decoder."""
        self.mel_lengths = mel_lengths
        self._batch_size = tf.shape(targets)[0]
        self.targets = targets[:, self.config.reduction_factor - 1::self.config.reduction_factor, :]
        self.max_lengths = tf.tile([tf.shape(self.targets)[1]], [self._batch_size])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    @property
    def reduction_factor(self):
        return self.config.reduction_factor

    def initialize(self):
        """Return (Finished, next_inputs)"""
        return (tf.tile([False], [self._batch_size]),
                tf.tile([[0.0]], [self._batch_size, self.config.n_mels]))

    def sample(self, time, outputs, state):
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, **kwargs):
        finished = (time + 1 >= self.max_lengths)
        next_inputs = self._ratio * self.targets[:, time, :] + \
            (1.0 - self._ratio) * outputs[:, -self.config.n_mels:]
        next_state = state
        return (finished, next_inputs, next_state)


class GmmAttention(AttentionMechanism):
    """Tensorflow GMM attention module."""

    def __init__(self,
                 config,
                 memory,
                 num_mixtures=8,
                 memory_sequence_length=None,
                 check_inner_dims_defined=True,
                 score_mask_value=None,
                 name='GmmAttention'):
        """Init variables."""
        self.config = config
        self.num_mixtures = num_mixtures
        self.query_layer = tf.keras.layers.Dense(
            3 * num_mixtures, name='gmm_query_layer', use_bias=True, dtype=tf.float32)

    @property
    def values(self):
        return self._value

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def alignments_size(self):
        return self._alignments_size

    @property
    def state_size(self):
        return self.num_mixtures

    def get_initial_state(self, batch_size, **kwargs):
        """Get initial alignments.
        Args:
            batch_size (int): batch_size of training.
        Returns:
            init kappa (float): kappa initial value, shape [batch_size, num_mixtures].
        """
        return tf.zeros(shape=[batch_size, self.num_mixtures], dtype=tf.float32)

    def get_initial_context(self, batch_size):
        """Get initial attention."""
        return tf.zeros(shape=[batch_size, self.config.encoder_lstm_units * 2], dtype=tf.float32)

    def setup_memory(self,
                     memory,
                     memory_sequence_length,
                     score_mask_value=None,
                     check_inner_dims_defined=True):
        if score_mask_value is None:
            score_mask_value = 0.0
        self._maybe_mask_score = partial(
            attention_wrapper._maybe_mask_score,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value)
        self._value = attention_wrapper._prepare_memory(
            memory, memory_sequence_length, memory_mask=None, check_inner_dims_defined=True)
        self._batch_size = (
            self._value.shape[0] or tf.shape(self._value)[0])
        self._alignments_size = (
            self._value.shape[1] or tf.shape(self._value)[1])
        self._keys = memory

    def __call__(self, inputs, training=False):
        """Call logic."""
        query, previous_kappa = inputs

        params = self.query_layer(query)

        alpha_hat, beta_hat, kappa_hat = tf.split(params, num_or_size_splits=3, axis=1)

        beta = tf.expand_dims(tf.exp(beta_hat), axis=2)
        kappa = tf.expand_dims(previous_kappa + tf.exp(kappa_hat), axis=2)

        # [batch_size, num_mixtures, 1]
        alpha = tf.expand_dims(tf.exp(alpha_hat), axis=2)

        # [1, 1, max_input_steps]
        mu = tf.reshape(tf.cast(tf.range(self.alignments_size), dtype=tf.float32),
                        shape=[1, 1, self.alignments_size])

        # [batch_size, max_input_steps]
        phi = tf.reduce_sum(alpha * tf.exp(-beta * (kappa - mu) ** 8.), axis=1)

        alignments = self._maybe_mask_score(phi)
        state = tf.squeeze(kappa, axis=2)

        expanded_alignments = tf.expand_dims(alignments, 2)
        context = tf.reduce_sum(expanded_alignments * self.values, 1)

        return context, alignments, state


class TFTacotronLocationSensitiveAttention(BahdanauAttention):
    """Tacotron-2 Location Sensitive Attention module."""

    def __init__(self,
                 config,
                 memory,
                 mask_encoder=True,
                 memory_sequence_length=None,
                 is_cumulate=True):
        """Init variables."""
        memory_length = memory_sequence_length if (mask_encoder is True) else None
        super().__init__(
            units=config.attention_dim,
            memory=memory,
            memory_sequence_length=memory_length,
            probability_fn="softmax",
            name="LocationSensitiveAttention"
        )
        self.location_convolution = tf.keras.layers.Conv1D(
            filters=config.attention_filters,
            kernel_size=config.attention_kernel,
            padding='same',
            name='location_conv'
        )
        self.location_layer = tf.keras.layers.Dense(units=config.attention_dim,
                                                    use_bias=False,
                                                    name='location_layer')

        self.v = tf.keras.layers.Dense(1, use_bias=True, name='scores_attention')
        self.config = config
        self.is_cumulate = is_cumulate

    def __call__(self, inputs, training=False):
        query, state = inputs

        processed_query = self.query_layer(query) if self.query_layer else query
        processed_query = tf.expand_dims(processed_query, 1)

        expanded_alignments = tf.expand_dims(state, axis=2)
        f = self.location_convolution(expanded_alignments)
        processed_location_features = self.location_layer(f)

        energy = self._location_sensitive_score(processed_query,
                                                processed_location_features,
                                                self.keys)

        alignments = self.probability_fn(energy, state)

        if self.is_cumulate:
            state = alignments + state
        else:
            state = alignments

        expanded_alignments = tf.expand_dims(alignments, 2)
        context = tf.reduce_sum(expanded_alignments * self.values, 1)

        return context, alignments, state

    def _location_sensitive_score(self, W_query, W_fil, W_keys):
        """Calculate location sensitive energy."""
        return tf.squeeze(self.v(tf.nn.tanh(W_keys + W_query + W_fil)), -1)

    def get_initial_state(self, batch_size, max_time):
        """Get initial alignments."""
        return tf.zeros(shape=[batch_size, max_time], dtype=tf.float32)

    def get_initial_context(self, batch_size):
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
    """Tacotron-2 postnet."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.conv_batch_norm = []
        for i in range(config.n_conv_postnet):
            conv = TFTacotronConvBatchNorm(
                filters=config.postnet_conv_filters,
                kernel_size=config.postnet_conv_kernel_sizes,
                dropout_rate=config.postnet_dropout_rate,
                activation='identity' if i + 1 == config.n_conv_postnet else 'tanh',
                name_idx=i
            )
            self.conv_batch_norm.append(conv)

    def call(self, inputs, training=False):
        """Call logic."""
        outputs = inputs
        for _, conv in enumerate(self.conv_batch_norm):
            outputs = conv(outputs)
        return outputs


TFTacotronDecoderCellState = collections.namedtuple(
    'TFTacotronDecoderCellState',
    ['attention_lstm_state', 'decoder_lstms_state', 'context', 'time', 'state', 'alignment_history'])

TFDecoderOutput = collections.namedtuple(
    "TFDecoderOutput", ("mel_output", "token_output", "sample_id"))


class TFTacotronDecoderCell(tf.keras.layers.AbstractRNNCell):
    """Tacotron-2 custom decoder cell."""

    def __init__(self, config, training, **kwargs):
        """Init variables."""
        super().__init__(**kwargs)
        self.training = training
        self.prenet = TFTacotronPrenet(config, name='prenet')

        # define lstm cell on decoder.
        # TODO(@dathudeptrai) switch to zone-out lstm.
        self.attention_lstm = tf.keras.layers.LSTMCell(units=config.decoder_lstm_units,
                                                       name='attention_lstm_cell')
        lstm_cells = []
        for i in range(config.n_lstm_decoder):
            lstm_cell = tf.keras.layers.LSTMCell(units=config.decoder_lstm_units,
                                                 name='lstm_cell_._{}'.format(i))
            lstm_cells.append(lstm_cell)
        self.decoder_lstms = tf.keras.layers.StackedRNNCells(lstm_cells,
                                                             name='decoder_lstms')

        # define attention layer.
        if config.attention_type == 'lsa':
            # create location-sensitive attention.
            self.attention_layer = TFTacotronLocationSensitiveAttention(
                config,
                memory=None,
                mask_encoder=True,
                memory_sequence_length=None,
                is_cumulate=True
            )
        elif config.attention_type == 'gmm':
            self.attention_layer = GmmAttention(
                config,
                memory=None,
                memory_sequence_length=None,
            )
        else:
            raise ValueError("Only lsa (location-sensitive attention) and gmm attention are supported")

        # frame, stop projection layer.
        self.frame_projection = tf.keras.layers.Dense(
            units=config.n_mels * config.reduction_factor, name='frame_projection')
        self.stop_projection = tf.keras.layers.Dense(units=config.reduction_factor, name='stop_projection')

        self.config = config

    def set_alignment_size(self, alignment_size):
        self.alignment_size = alignment_size

    @property
    def output_size(self):
        """Return output (mel) size."""
        return self.frame_projection.units

    @property
    def state_size(self):
        """Return hidden state size."""
        return TFTacotronDecoderCellState(
            attention_lstm_state=self.attention_lstm.state_size,
            decoder_lstms_state=self.decoder_lstms.state_size,
            time=tf.TensorShape([]),
            attention=self.config.attention_dim,
            state=self.alignment_size,
            alignment_history=(),
        )

    def get_initial_state(self, batch_size):
        """Get initial states."""
        initial_attention_lstm_cell_states = self.attention_lstm.get_initial_state(None, batch_size, dtype=tf.float32)
        initial_decoder_lstms_cell_states = self.decoder_lstms.get_initial_state(None, batch_size, dtype=tf.float32)
        initial_context = tf.zeros(shape=[batch_size, self.config.encoder_lstm_units * 2], dtype=tf.float32)
        initial_state = self.attention_layer.get_initial_state(batch_size, self.alignment_size)
        initial_alignment_history = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        return TFTacotronDecoderCellState(
            attention_lstm_state=initial_attention_lstm_cell_states,
            decoder_lstms_state=initial_decoder_lstms_cell_states,
            time=tf.zeros([], dtype=tf.int32),
            context=initial_context,
            state=initial_state,
            alignment_history=initial_alignment_history
        )

    def call(self, inputs, states):
        """Call logic."""
        decoder_input = inputs

        # 1. apply prenet for decoder_input.
        prenet_out = self.prenet(decoder_input, training=self.training)  # [batch_size, dim]

        # 2. concat prenet_out and prev context vector
        # then use it as input of attention lstm layer.
        attention_lstm_input = tf.concat([prenet_out, states.context], axis=-1)
        attention_lstm_output, next_attention_lstm_state = self.attention_lstm(
            attention_lstm_input, states.attention_lstm_state)

        # 3. compute context, alignment and cumulative alignment.
        prev_state = states.state
        prev_alignment_history = states.alignment_history
        context, alignments, state = self.attention_layer(
            [attention_lstm_output,
             prev_state],
            training=self.training
        )

        # 4. run decoder lstm(s)
        decoder_lstms_input = tf.concat([attention_lstm_output, context], axis=-1)
        decoder_lstms_output, next_decoder_lstms_state = self.decoder_lstms(
            decoder_lstms_input,
            states.decoder_lstms_state
        )

        # 4. compute frame feature and stop token.
        projection_inputs = tf.concat([decoder_lstms_output, context], axis=-1)
        decoder_outputs = self.frame_projection(projection_inputs)

        stop_inputs = tf.concat([decoder_lstms_output, decoder_outputs], axis=-1)
        stop_tokens = self.stop_projection(stop_inputs)

        # 5. save alignment history to visualize.
        alignment_history = prev_alignment_history.write(states.time, alignments)

        # 6. return new states.
        new_states = TFTacotronDecoderCellState(
            attention_lstm_state=next_attention_lstm_state,
            decoder_lstms_state=next_decoder_lstms_state,
            time=states.time + 1,
            context=context,
            state=state,
            alignment_history=alignment_history
        )

        return (decoder_outputs, stop_tokens), new_states


class TFTacotronDecoder(Decoder):
    """Tacotron-2 Decoder."""

    def __init__(self,
                 decoder_cell,
                 decoder_sampler,
                 output_layer=None):
        """Initial variables."""
        self.cell = decoder_cell
        self.sampler = decoder_sampler
        self.output_layer = output_layer

    def setup_decoder_init_state(self, decoder_init_state):
        self.initial_state = decoder_init_state

    def initialize(self, **kwargs):
        return self.sampler.initialize() + (self.initial_state,)

    @property
    def output_size(self):
        return TFDecoderOutput(
            mel_output=tf.nest.map_structure(
                lambda shape: tf.TensorShape(shape), self.cell.output_size),
            token_output=tf.TensorShape(self.sampler.reduction_factor),
            sample_id=self.sampler.sample_ids_shape
        )

    @property
    def output_dtype(self):
        return TFDecoderOutput(
            tf.float32,
            tf.float32,
            self.sampler.sample_ids_dtype
        )

    @property
    def batch_size(self):
        return self.sampler._batch_size

    def step(self, time, inputs, state, training=False):
        (mel_outputs, stop_tokens), cell_state = self.cell(inputs, state, training=training)
        if self.output_layer is not None:
            mel_outputs = self.output_layer(mel_outputs)
        sample_ids = self.sampler.sample(
            time=time, outputs=mel_outputs, state=cell_state
        )
        (finished, next_inputs, next_state) = self.sampler.next_inputs(
            time=time,
            outputs=mel_outputs,
            state=cell_state,
            sample_ids=sample_ids,
            stop_token_prediction=stop_tokens
        )

        outputs = TFDecoderOutput(mel_outputs, stop_tokens, sample_ids)
        return (outputs, next_state, next_inputs, finished)


class TFTacotron2(tf.keras.Model):
    """Tensorflow tacotron-2 model."""

    def __init__(self, config, training, **kwargs):
        """Initalize tacotron-2 layers."""
        super().__init__(self, **kwargs)
        self.encoder = TFTacotronEncoder(config, name='encoder')
        self.decoder_cell = TFTacotronDecoderCell(config, training=training, name='decoder_cell')
        self.decoder = TFTacotronDecoder(
            self.decoder_cell,
            TrainingSampler(config)
        )
        self.postnet = TFTacotronPostnet(config, name='post_net')
        self.post_projection = tf.keras.layers.Dense(units=config.n_mels,
                                                     name='residual_projection')

        self.config = config

    def _build(self):
        input_ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        input_lengths = np.array([9])
        speaker_ids = np.array([0])
        mel_outputs = np.random.normal(size=(1, 50, 80))
        mel_lengths = np.array([50])
        self(input_ids, input_lengths, speaker_ids, mel_outputs, mel_lengths, mel_lengths, training=True)

    def call(self,
             input_ids,
             input_lengths,
             speaker_ids,
             mel_outputs,
             mel_lengths,
             real_lengths=None,
             training=False):
        """Call logic."""
        # create input-mask based on input_lengths
        input_mask = tf.sequence_mask(input_lengths,
                                      maxlen=tf.reduce_max(input_lengths),
                                      name='input_sequence_masks')

        # Encoder Step.
        encoder_hidden_states = self.encoder([input_ids, speaker_ids, input_mask], training=training)

        batch_size = tf.shape(encoder_hidden_states)[0]
        alignment_size = tf.shape(encoder_hidden_states)[1]

        # Setup some initial placeholders for decoder step. Include:
        # 1. mel_outputs, mel_lengths for teacher forcing mode.
        # 2. alignment_size for attention size.
        # 3. initial state for decoder cell.
        # 4. memory (encoder hidden state) for attention mechanism.
        self.decoder.sampler.setup_target(targets=mel_outputs, mel_lengths=mel_lengths)
        self.decoder.cell.set_alignment_size(alignment_size)
        self.decoder.setup_decoder_init_state(
            self.decoder.cell.get_initial_state(batch_size)
        )
        self.decoder.cell.attention_layer.setup_memory(
            memory=encoder_hidden_states,
            memory_sequence_length=input_lengths  # use for mask attention.
        )

        # run decode step.
        (frames_prediction, stop_token_prediction, _), final_decoder_state, _ = dynamic_decode(
            self.decoder
        )

        decoder_output = tf.reshape(frames_prediction, [batch_size, -1, self.config.n_mels])
        stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])

        residual = self.postnet(decoder_output, training=training)
        residual_projection = self.post_projection(residual)

        mel_outputs = decoder_output + residual_projection

        alignment_history = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

        return decoder_output, mel_outputs, stop_token_prediction, alignment_history
