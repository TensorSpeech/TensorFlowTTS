# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen (@dathudeptrai) - Eren Gölge (@erogol)
#  MIT License (https://opensource.org/licenses/MIT)

"""Tacotron-2 Modules."""

import tensorflow as tf

from tensorflow_tts.layers import TFTacotronEncoder
from tensorflow_tts.layers import TFTacotronLocationSensitiveAttention
from tensorflow_tts.layers import TFTacotronDecoderCell
from tensorflow_tts.layers import TFTacotronDecoderInput
from tensorflow_tts.layers import TFTacotronPostnet

from tensorflow_tts.configs import Tacotron2Config


class TFTacotron2(tf.keras.Model):
    """Tensorflow tacotron-2 model."""

    def __init__(self, config, training, **kwargs):
        """Initalize tacotron-2 layers."""
        super().__init__(self, **kwargs)
        self.encoder = TFTacotronEncoder(config, name='encoder')
        self.decoder_cell = TFTacotronDecoderCell(config, training=training, name='decoder_cell')
        self.postnet = TFTacotronPostnet(config, name='post_net')
        self.post_projection = tf.keras.layers.Dense(units=config.n_mels,
                                                     name='residual_projection')

    # @tf.function
    def call(self,
             input_ids,
             speaker_ids,
             mel_outputs,
             input_lengths,
             mel_lengths,
             training=False):
        """Call logic."""
        # create input-mask based on input_lengths
        max_length = tf.reduce_max(input_lengths)
        input_mask = tf.cast(tf.sequence_mask(input_lengths, maxlen=max_length), tf.float32)  # [batch_size, max_length]
        encoder_hidden_states = self.encoder([input_ids, speaker_ids, input_mask], training=training)

        batch_size, max_length_encoder = tf.keras.backend.int_shape(encoder_hidden_states)[0:2]

        # decoder
        max_decoder_steps = tf.reduce_max(mel_lengths)
        time_first_mels_outputs = tf.transpose(mel_outputs, perm=[1, 0, 2])  # [max_len, batch_size, dim]

        frame_predictions = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        stop_predictions = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        state = self.decoder_cell.get_initial_state(
            batch_size=batch_size,
            alignment_size=max_length_encoder
        )
        num_step = tf.constant(0, dtype=tf.int32)
        for _ in tf.range(max_decoder_steps):
            decoder_inputs = TFTacotronDecoderInput(
                time_first_mels_outputs[num_step],
                encoder_hidden_states,
                input_mask
            )
            outputs, state = self.decoder_cell(decoder_inputs, state)
            frame_pred, stop_pred = outputs
            frame_predictions = frame_predictions.write(num_step, frame_pred)
            stop_predictions = stop_predictions.write(num_step, stop_pred)
            num_step += 1

        mel_outputs = tf.transpose(frame_predictions.stack(), [1, 0, 2])
        stop_outputs = tf.transpose(stop_predictions.stack(), [1, 0, 2])

        # calculate decoder mask
        mel_mask = tf.cast(tf.sequence_mask(mel_lengths, max_decoder_steps), tf.float32)
        residual = self.postnet([mel_outputs, mel_mask], training=training)
        residual_projection = self.post_projection(residual)
        post_mel_outputs = mel_outputs + residual_projection

        alignment_history = tf.transpose(state.alignment_history.stack(), [1, 0, 2])

        return mel_outputs, post_mel_outputs, stop_outputs, alignment_history
