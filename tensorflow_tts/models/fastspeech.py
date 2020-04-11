# -*- coding: utf-8 -*-

# Copyright 2020 MINH ANH (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

"""Tensorflow Model modules for FastSpeech."""

import tensorflow as tf

from tensorflow_tts.layers import TFFastSpeechEmbeddings
from tensorflow_tts.layers import TFFastSpeechEncoder
from tensorflow_tts.layers import TFFastSpeechDurationPredictor
from tensorflow_tts.layers import TFFastSpeechLengthRegulator


class TFFastSpeech(tf.keras.Model):
    """TF Fastspeech module."""

    def __init__(self, config, **kwargs):
        """Init layers for fastspeech."""
        super().__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers

        self.embeddings = TFFastSpeechEmbeddings(config, name='embeddings')
        self.encoder = TFFastSpeechEncoder(config, name='encoder')
        self.duration_predictor = TFFastSpeechDurationPredictor(config, name='duration_predictor')
        self.length_regulator = TFFastSpeechLengthRegulator(config, name='length_regulator')
        self.decoder = TFFastSpeechEncoder(config, name='decoder')
        self.mels_dense = tf.keras.layers.Dense(units=config.num_mels)

    def call(
            self,
            input_ids,
            attention_mask,
            speaker_ids,
            duration_gts,
            training=False):
        """Call logic."""
        # extended_attention_masks for self attention encoder.
        extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings([input_ids, speaker_ids], training=training)
        encoder_output = self.encoder([embedding_output, extended_attention_mask], training=training)
        last_encoder_hidden_states = encoder_output[0]
        length_regulator_outputs, encoder_masks = self.length_regulator([
            last_encoder_hidden_states, duration_gts], training=training)

        # extend_encoder_masks for self attention decoder.
        extended_encoder_mask = encoder_masks[:, tf.newaxis, tf.newaxis, :]
        extended_encoder_mask = tf.cast(extended_encoder_mask, tf.float32)
        extended_encoder_mask = (1.0 - extended_encoder_mask) * -10000.0
        decoder_output = self.decoder([length_regulator_outputs, extended_encoder_mask], training=training)
        last_decoder_hidden_states = decoder_output[0]

        # here u can use sum or concat more than 1 hidden states layers from decoder.
        mel_outputs = self.mels_dense(last_decoder_hidden_states)

        # duration predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for duration_predictor.
        duration_outputs = self.duration_predictor(last_encoder_hidden_states)  # [batch_size, length]

        # mask duration outputs to force all padding charactor have duration = 0.
        masked_duration_outputs = duration_outputs * tf.cast(attention_mask, tf.float32)

        # mask mel outputs for force all padding mels have 0 value.
        masked_mel_outputs = mel_outputs * encoder_masks[:, :, tf.newaxis]

        outputs = (masked_mel_outputs, masked_duration_outputs)
        return outputs
