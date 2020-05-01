# -*- coding: utf-8 -*-

# Copyright 2020 MINH ANH (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

"""Tensorflow Model modules for FastSpeech."""

import tensorflow as tf

from tensorflow_tts.layers import TFFastSpeechEmbeddings
from tensorflow_tts.layers import TFFastSpeechEncoder
from tensorflow_tts.layers import TFFastSpeechDecoder
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
        #self.mels_dense = tf.keras.layers.Dense(units=config.num_mels)
        self.mels_dense = tf.keras.layers.Conv1D(filters=config.num_mels, kernel_size=3, padding='same')

        # build model.
        self._build()

    def _build(self):
        """Dummy input for building model."""
        # fake inputs
        input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        attention_mask = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        duration_gts = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
        self(input_ids, attention_mask, speaker_ids, duration_gts, training=True)

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
        mel_outputs = self.mels_dense(last_decoder_hidden_states)

        outputs = (mel_outputs, duration_outputs)
        return outputs

    def inference(self,
                  input_ids,
                  attention_mask,
                  speaker_ids,
                  duration_gts=None,
                  training=False):
        """Call logic."""
        embedding_output = self.embeddings([input_ids, speaker_ids], training=training)
        encoder_output = self.encoder([embedding_output, attention_mask], training=training)
        last_encoder_hidden_states = encoder_output[0]

        # duration predictor, here use last_encoder_hidden_states, u can use more hidden_states layers
        # rather than just use last_hidden_states of encoder for duration_predictor.
        duration_outputs = self.duration_predictor([last_encoder_hidden_states, attention_mask])  # [batch_size, length]
        duration_outputs = tf.math.exp(duration_outputs) - 1
        duration_outputs = tf.cast(tf.math.round(duration_outputs), tf.int32)

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
        mel_outputs = self.mels_dense(last_decoder_hidden_states)

        outputs = (mel_outputs, duration_outputs)
        return outputs

# if __name__ == "__main__":
#     from tensorflow_tts.configs.fastspeech import FastSpeechConfig
#     config = FastSpeechConfig()
#     fastspeech = TFFastSpeech(config=config)
#     fastspeech._build()

#     fastspeech.load_weights('./model-70000.h5')

#     # inference
#     import numpy as np
#     ids = np.load('LJ001-0009-ids.npy')
#     ids = np.expand_dims(ids, 0)

#     duration_gts = np.load('LJ001-0009-durations.npy')
#     duration_gts = np.expand_dims(duration_gts, 0)

#     mel, duration = fastspeech.inference(ids, tf.math.not_equal(ids, 0),
#                 speaker_ids=np.array([0]), duration_gts=duration_gts, training=False)
#     mel = mel[0].numpy()

#     # scaler mel
#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     scaler.mean_ = np.load("stats_new.npy")[0]
#     scaler.scale_ = np.load("stats_new.npy")[1]
#     mel_inverse = scaler.inverse_transform(mel)

#     mel_inverse = mel_inverse.astype(np.float32)
#     # plot
#     mel_gt = tf.reshape(np.load("LJ001-0009-raw-feats.npy"), (-1, 80)).numpy()  # [length, 80]
#     mel_pred = tf.reshape(mel_inverse, (-1, 80)).numpy()  # [length, 80]

#     # plit figure and save it
#     import matplotlib.pyplot as plt
#     figname = './test.png'
#     fig = plt.figure(figsize=(10, 8))
#     ax1 = fig.add_subplot(311)
#     ax2 = fig.add_subplot(312)
#     im = ax1.imshow(np.rot90(mel_gt), aspect='auto', interpolation='none')
#     ax1.set_title('Target Mel-Spectrogram')
#     fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
#     ax2.set_title('Predicted Mel-Spectrogram')
#     im = ax2.imshow(np.rot90(mel_pred), aspect='auto', interpolation='none')
#     fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax2)
#     plt.tight_layout()
#     plt.savefig(figname)
#     plt.close()
#     # melgan generator
#     from tensorflow_tts.models import TFMelGANGenerator
#     from tensorflow_tts.configs import MelGANGeneratorConfig
#     import soundfile as sf

#     config = MelGANGeneratorConfig(filters=512)
#     melgan = TFMelGANGenerator(config=config, name='melgan_generator')

#     audio = melgan(np.expand_dims(np.load("LJ050-0233-feats.npy"), 0))
#     melgan.load_weights('generator-2160000.h5')

#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     scaler.mean_ = np.load("stats.npy")[0]
#     scaler.scale_ = np.load("stats.npy")[1]
#     mel = scaler.transform(mel_inverse)

#     #norm_mel = scaler.transform(mel)
#     audio_pred = melgan(np.expand_dims(mel, 0))[0, :, 0]
#     sf.write('./test.wav', audio_pred,
#                 22050, "PCM_16")

    # decoder_model.load_weights('./test.h5')

    # fastspeech = TFFastSpeech(config=config, name='fastspeech')

    # input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 0, 0, 0],
    #                                   [1, 2, 3, 4, 5, 6, 7, 1, 1, 0]], tf.int32)
    # attention_mask = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    #                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]], tf.int32)
    # speaker_ids = tf.convert_to_tensor([0, 0], tf.int32)
    # duration_gts = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    #                                      [1, 1, 1, 1, 1, 1, 1, 5, 2, 0]], tf.int32)
    # outputs = fastspeech(input_ids, attention_mask, speaker_ids, duration_gts)

    # fastspeech.summary()
