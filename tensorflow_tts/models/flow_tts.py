# -*- coding: utf-8 -*-
# Copyright 2020 Mokke Meguru (@MokkeMeguru) Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flow-TTS model."""
from typing import Dict

import tensorflow as tf
from TFGENZOO.flows import AffineCouplingMask
from TFGENZOO.flows.cond_affine_coupling import (ConditionalAffineCoupling,
                                                 filter_kwargs)
from TFGENZOO.flows.flowbase import ConditionalFlowModule
from TFGENZOO.flows.flowbase import FactorOutBase
from TFGENZOO.flows.flowbase import FlowComponent
from TFGENZOO.flows.inv1x1conv import regular_matrix_init
from TFGENZOO.flows.utils import gaussianize
from TFGENZOO.flows.utils.util import split_feature


class Squeeze2D(FlowComponent):
    """Squeeze2D module."""

    def __init__(self, with_zaux: bool = False):
        self.with_zaux = with_zaux
        super().__init__()

    def get_config(self):
        config = super().get_config()
        config_update = {"with_zaux": self.with_zaux}
        config.update(config_update)
        return config

    def forward(self,
                x: tf.Tensor,
                zaux: tf.Tensor = None,
                mask: tf.Tensor = None,
                **kwargs):
        """ Forward logic.
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

    def inverse(self,
                z: tf.Tensor,
                zaux: tf.Tensor = None,
                mask: tf.Tensor = None,
                **kwargs):
        """ Inverse logic.
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


class Inv1x1Conv2DWithMask(FlowComponent):
    """Inv1x1Conv2DWithMask module."""

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
        """ Forward logic.
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
        """Inverse Logic."""
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


class ConditionalAffineCouplingWithMask(ConditionalAffineCoupling):
    """Conditional Affine Coupling Layer with mask.

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
        """ Forward logic.
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
        """Inverse Logic."""
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


class GTU(tf.keras.layers.Layer):
    """GTU layer proposed in Flow-TTS.

    Notes:

        * formula
            .. math::
                z = tanh(W_{f, k} \star y) \odot sigmoid(W_{g, k} \star c)
    """

    def __init__(self, **kwargs):
        super().__init__()

    def build(self, input_shape: tf.TensorShape):

        self.conv_first = tf.keras.layers.Conv1D(
            input_shape[-1],
            kernel_size=1,
            strides=1,
            padding="same",
            data_format="channels_last",
        )
        self.conv_last = tf.keras.layers.Conv1D(
            input_shape[-1],
            kernel_size=1,
            strides=1,
            padding="same",
            data_format="channels_last",
        )

        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config_update = {}
        config.update(config_update)
        return config

    def call(self, y: tf.Tensor, c: tf.Tensor, **kwargs):
        """ Call logic.
        Args:

            y (tf.Tensor): input contents tensor [B, T, C]
            c (tf.Tensor): input conditional tensor [B, T, C'] where C' can be different with C

        Returns:

            tf.Tensor: [B, T, C]
        """

        right = tf.nn.tanh(self.conv_first(y))
        left = tf.nn.sigmoid(self.conv_last(c))
        z = right * left
        return z


def CouplingBlock(x: tf.Tensor, cond: tf.Tensor, depth, **kwargs):
    """ CouplingBlock module.
    Args:

        x (tf.Tensor): input contents tensor [B, T, C]
        c (tf.Tensor): input conditional tensor [B, T, C'] where C' can be different with C
    Returns:

        tf.keras.Model: CouplingBlock
                        reference: Flow-TTS
    Examples:

        >>> import tensorflow as tf
        >>> from utils.coupling_block import CouplingBlock
        >>> x = tf.keras.layers.Input([None, 32])
        >>> c = tf.keras.layers.Input([None, 128])
        >>> cp = CouplingBlock(x, c, depth=256)
        >>> cp.summary()
        Model: "model"
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to
        ==================================================================================================
        input_1 (InputLayer)            [(None, None, 32)]   0
        __________________________________________________________________________________________________
        conv1d (Conv1D)                 (None, None, 256)    8192        input_1[0][0]
        __________________________________________________________________________________________________
        input_2 (InputLayer)            [(None, None, 128)]  0
        __________________________________________________________________________________________________
        gtu (GTU)                       (None, None, 256)    98816       conv1d[0][0]
        __________________________________________________________________________________________________
        tf_op_layer_AddV2 (TensorFlowOp [(None, None, 256)]  0           gtu[0][0]
                                                                        conv1d[0][0]
        __________________________________________________________________________________________________
        conv1d_1 (Conv1D)               (None, None, 64)     16448       tf_op_layer_AddV2[0][0]
        ==================================================================================================
        Total params: 123,456
        Trainable params: 123,456
        Non-trainable params: 0
        __________________________________________________________________________________________________
        >>> cp([x, c])
        <tf.Tensor 'model_1/Identity:0' shape=(None, None, 64) dtype=float32>
    """
    last_channel = x.shape[-1]
    c = cond

    conv1x1_1 = tf.keras.layers.Conv1D(
        depth,
        kernel_size=1,
        strides=1,
        padding="same",
        data_format="channels_last",
        use_bias=False,
        activation="relu",
    )

    gtu = GTU()

    conv1x1_2 = tf.keras.layers.Conv1D(
        last_channel * 2,
        kernel_size=1,
        strides=1,
        padding="same",
        data_format="channels_last",
        kernel_initializer="zeros",
        use_bias=False,
    )

    y = x
    y = conv1x1_1(y)
    z = gtu(y, c)
    y = tf.keras.layers.Add()([z, y])
    y = conv1x1_2(y)
    model = tf.keras.Model([x, c], y)
    return model


class Conv1DZeros(tf.keras.layers.Layer):
    """"Conv1DZeros module."""

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
    """Basic Factor Out Layer.

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
            x = x[..., self.split_size:]

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
            **kwargs):
        if zaux is not None:
            new_z = zaux[..., -self.split_size:]
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


def build_flow_step(
    step_num: int,
    coupling_depth: int,
    conditional_input: tf.keras.layers.Input,
    scale_type: str = "safe_exp",
):
    """utility function to construct step-of-flow.

    Sources:

        Flow-TTS's Figure 1

    Args:
        step_num       (int): K in Flow-TTS's Figure 1 (a).
            Number of flow-step iterations
        coupling_depth (int): coupling block's depth
        conditional_input (tf.keras.layers.Input): conditional Input Tensor
        scale_type     (str): Affine Coupling scale function: log_scale -> scale

    Returns:
        ConditionalFlowModule: flow-step's Module

    Examples:

    """

    def CouplingBlockTemplate(x: tf.Tensor):
        cb = CouplingBlock(x, cond=conditional_input, depth=coupling_depth)
        return cb

    cfml = []
    for i in range(step_num):

        # Sources:
        #
        #    FLow-TTS's Figure 1 (b)

        # Inv1x1Conv
        inv1x1 = Inv1x1Conv2DWithMask()

        # CouplingBlock
        couplingBlockTemplate = CouplingBlockTemplate

        # Affine_xform + Coupling Block
        #
        # Notes:
        #
        #     * forward formula
        #         |
        #         |  where x is source input [B, T, C]
        #         |        c is conditional input [B, T, C'] where C' can be difference with C
        #         |
        #         |  x_1, x_2 = split(x)
        #         |  z_1 = x_1
        #         |  [logs, shift] = NN(x_1, c)
        #         |  z_2 = (x_2 + shift) * exp(logs)
        #    * Coupling Block formula
        #         |
        #         |  where x_1', x_1'' is [B, T, C''] where C'' can be difference with C and C'
        #         |        logs, shift is [B, T, C]
        #         |
        #         |  x_1' =  1x1Conv_1(x_1)
        #         |  x_1'' = GTU(x_1', c)
        #         |  [logs, shift] = 1x1Conv_2(x_1'' + x')
        #         |
        #         |  GTU(x_1, c) = tanh(W_{f, k} * x_1) \odot \sigma(W_{g, k} * c)
        #         |  where W_{f, k} and W_{g, k} are 1-D convolution
        #
        conditionalAffineCoupling = ConditionalAffineCouplingWithMask(
            scale_shift_net_template=couplingBlockTemplate, scale_type=scale_type
        )
        cfml.append(inv1x1)
        cfml.append(conditionalAffineCoupling)
    return ConditionalFlowModule(cfml)


class FlowTTSDecoder(tf.keras.Model):
    """FlowTTSDecoder module."""

    def __init__(self, hparams: Dict, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.build_model()

    def build_model(self):
        conditionalInput = tf.keras.layers.Input(
            [None, self.hparams["conditional_width"]]
        )

        squeeze = Squeeze2D()

        flow_step_1 = build_flow_step(
            step_num=self.hparams["flow_step_depth"],
            coupling_depth=self.hparams["conditional_depth"],
            conditional_input=conditionalInput,
            scale_type=self.hparams["scale_type"],
        )

        factor_out_1 = FactorOutWithMask(
            with_zaux=False, conditional=self.hparams["conditional_factor_out"]
        )

        flow_step_2 = build_flow_step(
            step_num=self.hparams["flow_step_depth"],
            coupling_depth=self.hparams["conditional_depth"],
            conditional_input=conditionalInput,
            scale_type=self.hparams["scale_type"],
        )

        factor_out_2 = FactorOutWithMask(
            with_zaux=True, conditional=self.hparams["conditional_factor_out"]
        )

        flow_step_3 = build_flow_step(
            step_num=self.hparams["last_flow_step_depth"],
            coupling_depth=self.hparams["conditional_depth"],
            conditional_input=conditionalInput,
            scale_type=self.hparams["scale_type"],
        )

        self.flows = [
            squeeze,
            flow_step_1,
            factor_out_1,
            flow_step_2,
            factor_out_2,
            flow_step_3,
        ]

    def call(
            self,
            x: tf.Tensor,
            cond: tf.Tensor,
            zaux: tf.Tensor = None,
            mask: tf.Tensor = None,
            inverse: bool = False,
            training: bool = True,
            temparature: float = 1.0,
            **kwargs):
        """ Call logic.
        Args:
           x       (tf.Tensor): base input tensor [B, T, C]
           cond    (tf.Tensor): conditional input tensor [B, T, C']
           mask    (tf.Tensor): tensor has sequence length information [B, T]
           inverse      (bool): the flag of the invertible network
           training     (bool): training flag
           temparature (float): sampling temparature

        Notes:
            * forward returns
                - z                (tf.Tensor) [B, T, C_1]
                - log_det_jacobian (tf.Tensor) [B]
                - zaux             (tf.Tensor) [B, T, C_2] where C = C_1 + C_2
                - log_likelihood   (tf.Tensor) [B]
           * inverse returns
                - x                        (tf.Tensor) [B, T, C_1]
                - inverse_log_det_jacobian (tf.Tensor) [B]
        """
        if inverse:
            return self.inverse(
                x,
                cond=cond,
                zaux=zaux,
                mask=mask,
                training=training,
                temparature=temparature,
                **kwargs
            )
        else:
            return self.forward(x, cond=cond, training=training, mask=mask, **kwargs)

    def inverse(
        self,
        x: tf.Tensor,
        cond: tf.Tensor,
        zaux: tf.Tensor,
        training: bool,
        mask: tf.Tensor,
        temparature: float,
        **kwargs
    ):
        """inverse function.
        latent -> object
        """
        inverse_log_det_jacobian = tf.zeros(tf.shape(x)[0:1])

        for flow in reversed(self.flows):
            if isinstance(flow, Squeeze2D):
                x, zaux = flow(x, zaux=zaux, mask=mask, inverse=True)
                if mask is not None:
                    _, t = mask.shape
                    mask = tf.reshape(
                        tf.tile(tf.expand_dims(mask, -1), [1, 1, 2]), [-1, t * 2]
                    )

            elif isinstance(flow, FactorOutBase):
                if flow.with_zaux:
                    x, zaux = flow(
                        x, zaux=zaux, inverse=True, mask=mask, temparature=temparature
                    )
                else:
                    x = flow(
                        x, zaux=zaux, inverse=True, mask=mask, temparature=temparature
                    )
            else:
                x, ildj = flow(x, cond=cond, inverse=True, training=training, mask=mask)
                inverse_log_det_jacobian += ildj
        return x, inverse_log_det_jacobian

    def forward(self, x: tf.Tensor, cond: tf.Tensor, training: bool, mask: tf.Tensor, **kwargs):
        """forward function.
        object -> latent
        """
        zaux = None
        log_det_jacobian = tf.zeros(tf.shape(x)[0:1])
        log_likelihood = tf.zeros(tf.shape(x)[0:1])
        for flow in self.flows:
            if isinstance(flow, Squeeze2D):
                if flow.with_zaux:
                    x, zaux = flow(x, zaux=zaux)
                else:
                    x = flow(x)
                _, t = mask.shape
                mask = tf.reshape(mask, [-1, t // 2, 2])[..., 0]
            elif isinstance(flow, FactorOutBase):
                if x is None:
                    raise Exception()
                x, zaux, ll = flow(x, zaux=zaux, mask=mask)
                log_likelihood += ll
            else:
                x, ldj = flow(x, cond=cond, training=training, mask=mask)
                log_det_jacobian += ldj
        return x, log_det_jacobian, zaux, log_likelihood
