#!/usr/bin/env python3

import tensorflow as tf


class GTU(tf.keras.layers.Layer):
    """GTU layer proposed in Flow-TTS

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
        """
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
    """
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
