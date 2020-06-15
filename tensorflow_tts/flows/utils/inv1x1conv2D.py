#!/usr/bin/env python3

from typing import Tuple

import numpy as np
import tensorflow as tf

from TFGENZOO.flows.inv1x1conv import regular_matrix_init
from TFGENZOO.flows.flowbase import FlowComponent


class Inv1x1Conv2DWithMask(FlowComponent):
    def __init__(self, **kwargs):
        super().__init__()

    def build(self, input_shape: tf.TensorShape):
        _, t, c = input_shape
        self.c = c
        self.W = self.add_weight(
            name="W",
            shape=(c, c),
            regularizer=tf.keras.regularizers.l2(0.01),
            initializer=regular_matrix_init,
        )
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config_update = {}
        config.update(config_update)
        return config

    def forward(self, x: tf.Tensor, mask: tf.Tensor = None, **kwargs):
        """
        Args:
            x    (tf.Tensor): base input tensor [B, T, C]
            mask (tf.Tensor): mask input tensor [B, T]

        Returns:
            z    (tf.Tensor): latent variable tensor [B, T, C]
            ldj  (tf.Tensor): log det jacobian [B]

        Notes:
            * mask's example
                | [[True, True, True, False],
                |  [True, False, False, False],
                |  [True, True, True, True],
                |  [True, True, True, True]]
        """
        _, t, _ = x.shape
        W = self.W + tf.eye(self.c) * 1e-5
        _W = tf.reshape(W, [1, self.c, self.c])
        z = tf.nn.conv1d(x, _W, [1, 1, 1], "SAME")

        # scalar
        # tf.math.log(tf.abs(tf.linalg.det(W))) == tf.linalg.slogdet(W)[1]
        log_det_jacobian = tf.cast(
            tf.linalg.slogdet(tf.cast(W, tf.float64))[1], tf.float32,
        )

        # expand as batch
        if mask is not None:
            # mask -> mask_tensor: [B, T] -> [B, T, 1]
            mask_tensor = tf.expand_dims(tf.cast(mask, tf.float32), [-1])
            z = z * mask_tensor
            log_det_jacobian = log_det_jacobian * tf.reduce_sum(
                tf.cast(mask, tf.float32), axis=[-1]
            )
        else:
            log_det_jacobian = tf.broadcast_to(log_det_jacobian * t, tf.shape(x)[0:1])
        return z, log_det_jacobian

    def inverse(self, z: tf.Tensor, mask: tf.Tensor = None, **kwargs):
        _, t, _ = z.shape
        W = self.W + tf.eye(self.c) * 1e-5
        _W = tf.reshape(tf.linalg.inv(W), [1, self.c, self.c])
        x = tf.nn.conv1d(z, _W, [1, 1, 1], "SAME")

        inverse_log_det_jacobian = tf.cast(
            -1 * tf.linalg.slogdet(tf.cast(W, tf.float64))[1], tf.float32,
        )

        if mask is not None:
            # mask -> mask_tensor: [B, T] -> [B, T, 1]
            mask_tensor = tf.expand_dims(tf.cast(mask, tf.float32), [-1])
            x = x * mask_tensor
            inverse_log_det_jacobian = inverse_log_det_jacobian * tf.reduce_sum(
                tf.cast(mask, tf.float32), axis=[-1]
            )
        else:
            inverse_log_det_jacobian = tf.broadcast_to(
                inverse_log_det_jacobian * t, tf.shape(z)[0:1]
            )
        return x, inverse_log_det_jacobian


if __name__ == "__main__":
    inv1x1conv2D = Inv1x1Conv2DWithMask()
    inv1x1conv2D.build((None, 32, 16))
    inputs = tf.keras.layers.Input([None, 16])
    model = tf.keras.Model(inputs, inv1x1conv2D(inputs))
    model.summary()
    y, ldj = model(tf.random.normal([128, 12, 16]))
    print(y.shape)
