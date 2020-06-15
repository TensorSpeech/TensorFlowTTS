#!/usr/bin/env python3

from TFGENZOO.flows.flowbase import FlowComponent
import tensorflow as tf


class Squeeze2D(FlowComponent):
    def __init__(self, with_zaux: bool = False):
        self.with_zaux = with_zaux
        super().__init__()

    def get_config(self):
        config = super().get_config()
        config_update = {"with_zaux": self.with_zaux}
        config.update(config_update)
        return config

    def forward(
        self, x: tf.Tensor, zaux: tf.Tensor = None, mask: tf.Tensor = None, **kwargs
    ):
        """
        Args:
            x     (tf.Tensor): input tensor [B, T, C]
            zaux  (tf.Tensor): pre-latent tensor [B, T, C'']
            mask  (tf.Tensor): mask tensor [B, T]
        Returns:
            tf.Tensor: reshaped input tensor [B, T // 2, C * 2]
            tf.Tensor: reshaped pre-latent tensor [B, T // 2, C'' * 2]
            tf.Tensor: reshaped mask tensor [B, T // 2]
        """
        _, t, c = x.shape
        z = tf.reshape(tf.reshape(x, [-1, t // 2, 2, c]), [-1, t // 2, c * 2])
        if zaux is not None:
            _, t, c = zaux.shape
            zaux = tf.reshape(tf.reshape(zaux, [-1, t // 2, 2, c]), [-1, t // 2, c * 2])
            return z, zaux
        else:
            return z

    def inverse(
        self, z: tf.Tensor, zaux: tf.Tensor = None, mask: tf.Tensor = None, **kwargs
    ):
        """
        Args:
            z    (tf.Tensor): input tensor [B, T // 2, C * 2]
            zaux (tf.Tensor): pre-latent tensor [B, T // 2, C'' * 2]
            mask (tf.Tensor): pre-latent tensor [B, T // 2]
        Returns:
            tf.Tensor: reshaped input tensor [B, T, C]
            tf.Tensor: reshaped pre-latent tensor [B, T, C'']
            tf.Tensor: mask tensor [B, T]
        """
        _, t, c = z.shape
        x = tf.reshape(tf.reshape(z, [-1, t, 2, c // 2]), [-1, t * 2, c // 2])
        if zaux is not None:
            _, t, c = zaux.shape
            zaux = tf.reshape(tf.reshape(zaux, [-1, t, 2, c // 2]), [-1, t * 2, c // 2])
            return x, zaux
        else:
            return x
