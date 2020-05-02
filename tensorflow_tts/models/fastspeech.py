# -*- coding: utf-8 -*-

# Copyright 2020 MINH ANH (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

"""Tensorflow Model modules for FastSpeech."""

import numpy as np
import tensorflow as tf

from tensorflow_tts.layers import TFFastSpeechEmbeddings
from tensorflow_tts.layers import TFFastSpeechEncoder
from tensorflow_tts.layers import TFFastSpeechDecoder
from tensorflow_tts.layers import TFTacotronPostnet
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
        self.decoder = TFFastSpeechDecoder(config, name='decoder')
        self.mel_dense = tf.keras.layers.Dense(units=config.num_mels, name='mel_before')
        self.postnet = TFTacotronPostnet(config=config, name='postnet')

        # build model.
        self._build()

    def _build(self):
        """Dummy input for building model."""
        # fake inputs
        input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        attention_mask = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        duration_gts = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
        self(input_ids, attention_mask, speaker_ids, duration_gts)

    def call(self,
             input_ids,
             attention_mask,
             speaker_ids,
             duration_gts,
             training=False):
        """Call logic."""
        embedding_output = self.embeddings([input_ids, speaker_ids], training=training)
        encoder_output = self.encoder([embedding_output, attention_mask], training=training)
        last_encoder_hidden_states = encoder_output[0]

        # duration predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for duration_predictor.
        duration_outputs = self.duration_predictor([last_encoder_hidden_states, attention_mask])  # [batch_size, length]

        length_regulator_outputs, encoder_masks = self.length_regulator([
            last_encoder_hidden_states, duration_gts], training=training)

        # create decoder positional embedding
        decoder_pos = tf.range(1, tf.shape(length_regulator_outputs)[1] + 1, dtype=tf.int32)
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * encoder_masks

        decoder_output = self.decoder(
            [length_regulator_outputs, encoder_masks, masked_decoder_pos], training=training)
        last_decoder_hidden_states = decoder_output[0]

        # here u can use sum or concat more than 1 hidden states layers from decoder.
        mel_before = self.mel_dense(last_decoder_hidden_states)
        mel_after = self.postnet([mel_before, encoder_masks], training=training) + mel_before

        outputs = (mel_before, mel_after, duration_outputs)
        return outputs

    def inference(self,
                  input_ids,
                  attention_mask,
                  speaker_ids,
                  duration_gts=None,
                  speed_ratios=None,
                  training=False):
        """Call logic."""
        embedding_output = self.embeddings([input_ids, speaker_ids], training=training)
        encoder_output = self.encoder([embedding_output, attention_mask], training=training)
        last_encoder_hidden_states = encoder_output[0]

        # duration predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for duration_predictor.
        duration_outputs = self.duration_predictor([last_encoder_hidden_states, attention_mask])  # [batch_size, length]
        duration_outputs = tf.math.exp(duration_outputs) - 1

        if speed_ratios is None:
            speed_ratios = tf.convert_to_tensor(np.array([1.0]))

        duration_outputs = tf.cast(tf.math.round(duration_outputs * speed_ratios), tf.int32)

        if duration_gts is not None:
            duration_outputs = duration_gts

        length_regulator_outputs, encoder_masks = self.length_regulator([
            last_encoder_hidden_states, duration_outputs], training=training)

        # create decoder positional embedding
        decoder_pos = tf.range(1, tf.shape(length_regulator_outputs)[1] + 1, dtype=tf.int32)
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * encoder_masks

        decoder_output = self.decoder(
            [length_regulator_outputs, encoder_masks, masked_decoder_pos], training=training)
        last_decoder_hidden_states = decoder_output[0]

        # here u can use sum or concat more than 1 hidden states layers from decoder.
        mel_before = self.mel_dense(last_decoder_hidden_states)
        mel_after = self.postnet([mel_before, encoder_masks], training=training) + mel_before

        outputs = (mel_before, mel_after, duration_outputs)
        return outputs
