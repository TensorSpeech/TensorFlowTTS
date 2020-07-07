import tensorflow as tf
import math

LOG_2_PI = math.log(2 * math.pi)


class MixDensityLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=False):
        """Call logic.

        Args:
            1. mel, Tensor (float32) shape [batch_size, mel_length, 80]
            2. encoder_hidden_state, Tensor (float32) shape [batch_size, char_length, 512]
        """
        mel, mu_sigma, mel_lengths, character_lengths = inputs
        max_mel_length = tf.reduce_max(mel_lengths)
        max_char_length = tf.reduce_max(character_lengths)
        batch_size = tf.shape(mel)[0]

        mu, log_sigma = tf.split(mu_sigma, 2, axis=-1)

        # I normalize mu from 0 to 4 as same as range of melspectrogram
        mu_expand = tf.expand_dims(tf.math.sigmoid(mu) * 4, axis=1)  # [batch_size, 1, max_char_length, 80]
        log_sigma_expand = tf.expand_dims(tf.math.tanh(log_sigma) * 4, axis=1)  # [batch_size, 1, char_length, 80]
        # mu_expand, log_sigma_expand: [batch_size, max_mel_length, max_char_length, 80]

        mel_expand = tf.expand_dims(mel, 2)  # [batch_size, mel_len, 1, 80]

        mel_minus_mu = mel_expand - mu_expand

        exponential = - 0.5 * tf.reduce_sum(mel_minus_mu * mel_minus_mu / tf.math.exp(2 * log_sigma_expand), axis=-1)
        log_prob = exponential - 0.5 * tf.reduce_sum(log_sigma_expand, axis=-1) - 0.5 * 80 * LOG_2_PI

        alphas = tf.transpose(tf.zeros_like(log_prob), [1, 0, 2])  # [mel_length, batch_size, char_length]
        alpha_init = tf.concat([tf.expand_dims(log_prob[:, 0, 0], axis=-1),
                                tf.tile([[-1e20]], [batch_size, max_char_length - 1])], axis=-1)
        alpha_init = tf.expand_dims(alpha_init, axis=0)

        alphas = tf.tensor_scatter_nd_update(alphas, [[0]], alpha_init)

        for i in tf.range(1, max_mel_length):
            previous_alpha = alphas[i - 1]  # [batch_size, char_length]
            previous_alpha_shift = tf.concat([tf.tile([[-1e20]], [batch_size, 1]),
                                              previous_alpha[:, :-1]], axis=-1)
            alpha = tf.reduce_logsumexp([previous_alpha, previous_alpha_shift], axis=0)
            alpha = alpha + log_prob[:, i, :]
            alpha = tf.expand_dims(alpha, axis=0)
            alphas = tf.tensor_scatter_nd_update(alphas, [[i]], alpha)

        alphas = tf.transpose(alphas, [1, 0, 2])
        batch_indices = tf.range(batch_size)[:, tf.newaxis]
        mel_indices = mel_lengths[:, tf.newaxis] - 1
        character_indices = character_lengths[:, tf.newaxis] - 1
        indices = tf.concat([batch_indices, mel_indices, character_indices], axis=-1)

        last_alpha = tf.gather_nd(alphas, indices)

        return log_prob, -tf.reduce_mean(last_alpha), alphas