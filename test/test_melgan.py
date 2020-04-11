# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen Quan Anh
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import os

import pytest
import tensorflow as tf

from tensorflow_tts.models import TFMelGANGenerator
from tensorflow_tts.models import TFMelGANMultiScaleDiscriminator

from tensorflow_tts.configs import MelGANGeneratorConfig
from tensorflow_tts.configs import MelGANDiscriminatorConfig

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")


def make_melgan_generator_args(**kwargs):
    defaults = dict(
        out_channels=1,
        kernel_size=7,
        filters=512,
        use_bias=True,
        upsample_scales=[8, 8, 2, 2],
        stack_kernel_size=3,
        stacks=3,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        padding_type="REFLECT"
    )
    defaults.update(kwargs)
    return defaults


def make_melgan_discriminator_args(**kwargs):
    defaults = dict(
        out_channels=1,
        scales=3,
        downsample_pooling='AveragePooling1D',
        downsample_pooling_params={
            "pool_size": 4,
            "strides": 2,
        },
        kernel_sizes=[5, 3],
        filters=16,
        max_downsample_filters=1024,
        use_bias=True,
        downsample_scales=[4, 4, 4, 4],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        padding_type="REFLECT"
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize(
    "dict_g, dict_d, dict_loss", [
        ({}, {}, {}),
        ({"kernel_size": 3}, {}, {}),
        ({"filters": 1024}, {}, {}),
        ({"stack_kernel_size": 5}, {}, {}),
        ({"stack_kernel_size": 5, "stacks": 2}, {}, {}),
        ({"upsample_scales": [4, 4, 4, 4]}, {}, {}),
        ({"upsample_scales": [8, 8, 2, 2]}, {}, {}),
        ({"filters": 1024, "upsample_scales": [8, 8, 2, 2]}, {}, {})
    ])
def test_melgan_trainable(dict_g, dict_d, dict_loss):
    batch_size = 4
    batch_length = 4096
    args_g = make_melgan_generator_args(**dict_g)
    args_d = make_melgan_discriminator_args(**dict_d)

    args_g = MelGANGeneratorConfig(**args_g)
    args_d = MelGANDiscriminatorConfig(**args_d)

    mels_spec = tf.random.uniform(shape=[batch_size, 4096 // 256, 80],
                                  dtype=tf.float32)
    y = tf.random.uniform(shape=[batch_size, batch_length, 1], minval=0, maxval=1,
                          dtype=tf.float32)

    generator = TFMelGANGenerator(args_g)
    discriminator = TFMelGANMultiScaleDiscriminator(args_d)

    optimizer_g = tf.keras.optimizers.Adam(lr=0.0001)
    optimizer_d = tf.keras.optimizers.Adam(lr=0.00005)

    @tf.function(experimental_relax_shapes=True)
    def train_g_step(mels, y):
        with tf.GradientTape() as g_tape:
            y_hat = generator(mels)  # [B, T, 1]
            p_hat = discriminator(y_hat)

            y, y_hat = tf.squeeze(y), tf.squeeze(y_hat)  # [B, T]
            adv_loss = 0.0
            for i in range(len(p_hat)):
                adv_loss += tf.keras.losses.MeanSquaredError()(
                    p_hat[i][-1], tf.ones(p_hat[i][-1].shape)
                )
            adv_loss /= (i + 1)
            p = discriminator(tf.expand_dims(y, 2))

            fm_loss = 0.0
            for i in range(len(p_hat)):
                for j in range(len(p_hat[i]) - 1):
                    fm_loss += tf.keras.losses.MeanAbsoluteError()(
                        p_hat[i][j], p[i][j]
                    )
            loss_g = adv_loss + 0.0 * fm_loss
        gradients = g_tape.gradient(loss_g, generator.trainable_variables)
        optimizer_g.apply_gradients(zip(gradients, generator.trainable_variables))
        tf.print("loss generator: ", loss_g)
        return y, y_hat

    def train_d_step(y, y_hat):
        with tf.GradientTape() as d_tape:
            y, y_hat = tf.expand_dims(y, 2), tf.expand_dims(y_hat, 2)  # [B, T]
            p = discriminator(y)
            p_hat = discriminator(y_hat)
            real_loss = 0.0
            fake_loss = 0.0
            for k in range(len(p)):
                real_loss += tf.keras.losses.MeanSquaredError()(
                    p[k][-1], tf.ones(p[k][-1].shape)
                )
                fake_loss += tf.keras.losses.MeanSquaredError()(
                    p_hat[k][-1], tf.zeros(p_hat[k][-1].shape)
                )
            real_loss /= (k + 1)
            fake_loss /= (k + 1)
            loss_d = real_loss + fake_loss
        gradients = d_tape.gradient(loss_d, discriminator.trainable_variables)
        optimizer_d.apply_gradients(zip(gradients, discriminator.trainable_variables))
        tf.print("loss discrimin: ", loss_d)

    for step in range(2):
        y, y_hat = train_g_step(mels_spec, y)

        if step >= 1:
            train_d_step(y, y_hat)
