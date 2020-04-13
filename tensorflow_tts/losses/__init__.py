# -*- coding: utf-8 -*-

# Copyright 2020 MINH ANH (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

"""Tensorflow losses."""

import tensorflow as tf

from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils


class TFReLuError(LossFunctionWrapper):
    """Computes the relu loss between true labels and predicted labels."""

    def __init__(self,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='relu_error'):
        super().__init__(
            relu_error, name=name, reduction=reduction)


def relu_error(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.keras.backend.mean(tf.maximum(y_true - y_pred, 0.0), axis=-1)


# aliasses
relu_error = melgan_relu_error = TFReLuError()
mae_error = tf.keras.losses.MeanAbsolutePercentageError()
