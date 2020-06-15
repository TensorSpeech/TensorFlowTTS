#!/usr/bin/env python3

import tensorflow as tf
from TFGENZOO.flows.cond_affine_coupling import ConditionalAffineCoupling, filter_kwargs
from TFGENZOO.flows import AffineCouplingMask


class ConditionalAffineCouplingWithMask(ConditionalAffineCoupling):
    """Conditional Affine Coupling Layer with mask

    Sources:
        https://github.com/masa-su/pixyz/blob/master/pixyz/flows/coupling.py

    Note:
        * forward formula
            | [x1, x2] = split(x)
            | log_scale, shift = NN([x1, c])
            | scale = exp(log_scale)
            | z1 = x1
            | z2 = (x2 + shift) * scale
            | z = concat([z1, z2])
            | LogDetJacobian = sum(log(scale))

        * inverse formula
            | [z1, z2] = split(x)
            | log_scale, shift = NN([z1, c])
            | scale = exp(log_scale)
            | x1 = z1
            | x2 = z2 / scale - shift
            | z = concat([x1, x2])
            | InverseLogDetJacobian = - sum(log(scale))

        * implementation notes
           | in Glow's Paper, scale is calculated by exp(log_scale),
           | but IN IMPLEMENTATION, scale is done by sigmoid(log_scale + 2.0)
           | where c is the conditional input for WaveGlow or cINN
           | https://arxiv.org/abs/1907.02392

        * TODO notes
           | cINN uses double coupling, but our coupling is single coupling
           |
           | scale > 0 because exp(x) > 0
    """

    def build(self, input_shape: tf.TensorShape):
        self.reduce_axis = list(range(len(input_shape)))[1:]
        if self.scale_shift_net is None:
            resnet_inputs = [None for _ in range(len(input_shape) - 1)]
            resnet_inputs[-1] = int(input_shape[-1] / 2)
            self.scale_shift_net = self.scale_shift_net_template(
                tf.keras.layers.Input(resnet_inputs)
            )
        super().build(input_shape)

    def forward(self, x: tf.Tensor, cond: tf.Tensor, mask: tf.Tensor = None, **kwargs):
        """
        Args:
            x    (tf.Tensor): base input tensor [B, T, C]
            cond (tf.Tensor): conditional input tensor [B, T, C']
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
        x1, x2 = tf.split(x, 2, axis=-1)
        z1 = x1
        h = self.scale_shift_net([x1, cond], **filter_kwargs(kwargs))
        if self.mask_type == AffineCouplingMask.ChannelWise:
            shift = h[..., 0::2]
            log_scale = h[..., 1::2]

            scale = self.scale_func(log_scale)

            # apply mask into scale, shift
            # mask -> mask_tensor: [B, T] -> [B, T, 1]
            if mask is not None:
                mask_tensor = tf.expand_dims(tf.cast(mask, tf.float32), [-1])
                scale *= mask_tensor
                shift *= mask_tensor
            z2 = (x2 + shift) * scale

            # scale's shape is [B, T, C]
            # log_det_jacobian's shape is [B]
            log_det_jacobian = tf.reduce_sum(tf.math.log(scale), axis=self.reduce_axis)
            return tf.concat([z1, z2], axis=-1), log_det_jacobian
        else:
            raise NotImplementedError()

    def inverse(self, z: tf.Tensor, cond: tf.Tensor, mask: tf.Tensor = None, **kwargs):
        z1, z2 = tf.split(z, 2, axis=-1)
        x1 = z1
        h = self.scale_shift_net([x1, cond], **filter_kwargs(kwargs))
        if self.mask_type == AffineCouplingMask.ChannelWise:
            shift = h[..., 0::2]
            log_scale = h[..., 1::2]

            scale = self.scale_func(log_scale)

            if mask is not None:
                mask_tensor = tf.expand_dims(tf.cast(mask, tf.float32), [-1])
                scale *= mask_tensor
                shift *= mask_tensor
            x2 = (z2 / scale) - shift

            inverse_log_det_jacobian = -1 * tf.reduce_sum(
                tf.math.log(scale), axis=self.reduce_axis
            )
            return tf.concat([x1, x2], axis=-1), inverse_log_det_jacobian
        else:
            raise NotImplementedError()
