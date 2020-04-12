# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen Quan Anh
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import os
import pytest
import numpy as np
import tensorflow as tf

from tensorflow_tts.models import TFTacotron2
from tensorflow_tts.configs import Tacotron2Config

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")


@pytest.mark.parametrize(
    "n_speakers", [
        (2), (3)
    ]
)
def test_tacotron2_trainable(n_speakers):
    config = Tacotron2Config(n_speakers=n_speakers)
    model = TFTacotron2(config, training=True)

    # fake input
    input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]], dtype=tf.int32)
    speaker_ids = tf.convert_to_tensor([0, 1], tf.int32)
    input_lengths = tf.convert_to_tensor([10, 5], tf.int32)
    mel_outputs_1 = tf.random.uniform(shape=[1, 50, 80])
    mel_outputs_2 = tf.pad(tf.random.uniform(shape=[1, 25, 80]), [[0, 0], [0, 25], [0, 0]])
    mel_outputs = tf.concat([mel_outputs_1, mel_outputs_2], axis=0)
    mel_lengths = [50, 25]

    stop_tokens = np.zeros((2, 50), np.float32)
    stop_tokens[0][49] = 1.0
    stop_tokens[1][24] = 1.0

    stop_tokens = tf.convert_to_tensor(stop_tokens)

    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    # @tf.function
    def one_step_training():
        with tf.GradientTape() as tape:
            mel_preds, \
                post_mel_preds, \
                stop_preds, \
                alignment_history = model(input_ids,
                                          speaker_ids,
                                          mel_outputs,
                                          input_lengths,
                                          mel_lengths, training=True)
            loss_before = tf.keras.losses.MeanSquaredError()(mel_outputs, mel_preds)
            loss_after = tf.keras.losses.MeanSquaredError()(mel_outputs, post_mel_preds)
            loss_stop_tokens = tf.keras.losses.BinaryCrossentropy(from_logits=True)(stop_tokens, stop_preds)
            loss = loss_before + loss_after + loss_stop_tokens
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        tf.print(loss_before)

    import time
    for i in range(10):
        if i == 1:
            start = time.time()
        one_step_training()
    print(time.time() - start)
