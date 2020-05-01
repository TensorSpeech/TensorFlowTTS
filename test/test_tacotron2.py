# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen Quan Anh - Eren Gölge
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import os
import pytest
import numpy as np
import tensorflow as tf

from tensorflow_tts.models import TFTacotron2
from tensorflow_tts.configs import Tacotron2Config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")


@pytest.mark.parametrize(
    "n_speakers, n_chars, max_input_length, max_mel_length, batch_size", [
        (2, 64, 25, 150, 32),
    ]
)
def test_tacotron2_trainable(n_speakers, n_chars, max_input_length, max_mel_length, batch_size):
    config = Tacotron2Config(n_speakers=n_speakers)
    model = TFTacotron2(config, training=True)

    # fake input
    input_ids = tf.random.uniform([batch_size, max_input_length], maxval=n_chars, dtype=tf.int32)
    input_lengths = np.random.randint(0, high=max_input_length+1, size=[batch_size])
    input_lengths[-1] = max_input_length
    input_lengths = tf.convert_to_tensor(input_lengths, dtype=tf.int32)
    speaker_ids = tf.convert_to_tensor([0] * batch_size, tf.int32)
    mel_outputs = tf.random.uniform(shape=[batch_size, max_mel_length, 80])
    mel_lengths =  np.random.randint(0, high=max_mel_length+1, size=[batch_size])
    mel_lengths[-1] = max_mel_length
    mel_lengths = tf.convert_to_tensor(mel_lengths, dtype=tf.int32)

    stop_tokens = np.zeros((batch_size, max_mel_length), np.float32)
    stop_tokens = tf.convert_to_tensor(stop_tokens)

    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    @tf.function
    def one_step_training(model, optimizer, input_ids, input_lengths, speaker_ids, mel_outputs, mel_lengths, stop_tokens):
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
        return loss

    import time
    for i in range(10):
        if i == 1:
            start = time.time()
        loss = one_step_training(model, optimizer, input_ids, input_lengths, speaker_ids, mel_outputs, mel_lengths, stop_tokens)
        print(f" > loss: {loss}")
    total_runtime = time.time() - start
    print(f" > Total run-time: {total_runtime}")
    print(f" > Avg run-time: {total_runtime/10}")

