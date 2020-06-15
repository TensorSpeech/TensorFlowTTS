#!/usr/bin/env python3

import tensorflow as tf
from TFGENZOO.flows.flowbase import FactorOutBase
from TFGENZOO.flows.utils import gaussianize
from TFGENZOO.flows.utils.util import split_feature
from TFGENZOO.flows.utils.conv_zeros import Conv2DZeros
from typing import Tuple


class Conv1DZeros(tf.keras.layers.Layer):
    def __init__(
        self,
        width_scale: int = 2,
        kernel_initializer="zeros",
        kernel_size: int = 3,
        logscale_factor: float = 3.0,
    ):
        super().__init__()
        self.width_scale = width_scale
        self.kernel_size = kernel_size
        self.initializer = kernel_initializer
        self.logscale_factor = logscale_factor

    def get_config(self):
        config = super().get_config()
        config_update = {
            "width_scale": self.width_scale,
            "kernel_initializer": tf.keras.initializers.serialize(
                tf.keras.initializers.get(self.initializer)
            ),
            "kernel_size": self.kernel_size,
        }
        config.update(config_update)
        return config

    def build(self, input_shape: tf.TensorShape):
        n_in = input_shape[-1]
        n_out = n_in * self.width_scale
        self.kernel = self.add_weight(
            name="kernel",
            initializer=self.initializer,
            shape=[self.kernel_size] + [n_in, n_out],
            dtype=tf.float32,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=[1 for _ in range(len(input_shape) - 1)] + [n_out],
            initializer="zeros",
        )
        self.logs = self.add_weight(
            name="logs", shape=[1, n_out], initializer=self.initializer
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor):
        x = tf.nn.conv1d(x, filters=self.kernel, stride=1, padding="SAME")
        x += self.bias
        x *= tf.exp(self.logs * self.logscale_factor)
        return x


class FactorOutWithMask(FactorOutBase):
    """Basic Factor Out Layer

    This layer drops factor-outed Tensor z_i

    Note:

        * forward procedure
           | input  : h_{i-1}
           | output : h_{i}, loss
           |
           | [z_i, h_i] = split(h_{i-1})
           |
           | loss =
           |     z_i \sim N(0, 1) if conditional is False
           |     z_i \sim N(mu, sigma) if conditional is True
           |  ,where
           | mu, sigma = Conv(h)

        * inverse procedure
           | input  : h_{i}
           | output : h_{i-1}
           |
           | sample z_i from N(0, 1) or N(mu, sigma) by conditional
           | h_{i-1} = [z_i, h_i]
    """

    def build(self, input_shape: tf.TensorShape):
        self.split_size = input_shape[-1] // 2
        super().build(input_shape)

    def __init__(self, with_zaux: bool = False, conditional: bool = False):
        super().__init__()
        self.with_zaux = with_zaux
        self.conditional = conditional
        if self.conditional:
            self.conv = Conv1DZeros(width_scale=2)

    def get_config(self):
        config = super().get_config()
        config_update = {}
        if self.conditional:
            config_update = {
                "conditional": self.conditional,
                "conv": self.conv.get_config(),
            }
        else:
            config_update = {"conditional": self.conditional}
        config.update(config_update)
        return config

    def split2d_prior(self, z: tf.Tensor):
        h = self.conv(z)
        return split_feature(h, "cross")

    def calc_ll(self, z1: tf.Tensor, z2: tf.Tensor, mask_tensor: tf.Tensor = None):
        """
        Args:
           z1 (tf.Tensor): [B, T, C // 2]
           z2 (tf.Tensor): [B, T, C // 2]
        """
        with tf.name_scope("calc_log_likelihood"):
            if self.conditional:
                mean, logsd = self.split2d_prior(z1)
                ll = gaussianize.gaussian_likelihood(mean, logsd, z2)
            else:
                ll = gaussianize.gaussian_likelihood(
                    tf.zeros(tf.shape(z2)), tf.zeros(tf.shape(z2)), z2
                )
            # ll is [B, T, C // 2]
            if mask_tensor is not None:
                ll *= mask_tensor
                ll = tf.reduce_sum(ll, axis=list(range(1, len(z2.shape))))
            return ll

    def forward(self, x: tf.Tensor, zaux: tf.Tensor = None, mask=None, **kwargs):
        if mask is not None:
            mask_tensor = tf.expand_dims(tf.cast(mask, tf.float32), axis=[-1])
        else:
            mask_tensor = None
        with tf.name_scope("split"):
            new_z = x[..., : self.split_size]
            x = x[..., self.split_size :]

        ll = self.calc_ll(x, new_z, mask_tensor=mask_tensor)

        if self.with_zaux:
            zaux = tf.concat([zaux, new_z], axis=-1)
        else:
            zaux = new_z
        return x, zaux, ll

    def inverse(
        self,
        z: tf.Tensor,
        zaux: tf.Tensor = None,
        mask=None,
        temparature: float = 0.2,
        **kwargs
    ):
        if zaux is not None:
            new_z = zaux[..., -self.split_size :]
            zaux = zaux[..., : -self.split_size]
        else:
            # TODO: sampling test
            mean, logsd = self.split2d_prior(z)
            new_z = gaussianize.gaussian_sample(mean, logsd, temparature)
        z = tf.concat([new_z, z], axis=-1)
        if self.with_zaux:
            return z, zaux
        else:
            return z
