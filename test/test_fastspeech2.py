# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
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

import logging
import os
import yaml

import pytest
import tensorflow as tf

from tensorflow_tts.configs import FastSpeech2Config
from tensorflow_tts.models import TFFastSpeech2
from tensorflow_tts.utils import return_strategy

from examples.fastspeech2.train_fastspeech2 import FastSpeech2Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


@pytest.mark.parametrize("new_size", [100, 200, 300])
def test_fastspeech_resize_positional_embeddings(new_size):
    config = FastSpeech2Config()
    fastspeech2 = TFFastSpeech2(config, name="fastspeech")
    fastspeech2._build()
    fastspeech2.save_weights("./test.h5")
    fastspeech2.resize_positional_embeddings(new_size)
    fastspeech2.load_weights("./test.h5", by_name=True, skip_mismatch=True)


@pytest.mark.parametrize(
    "var_train_expr, config_path",
    [
        (None, "./examples/fastspeech2/conf/fastspeech2.v1.yaml"),
        ("embeddings|encoder", "./examples/fastspeech2/conf/fastspeech2.v1.yaml"),
        ("embeddings|encoder", "./examples/fastspeech2/conf/fastspeech2.v2.yaml"),
        ("embeddings|encoder", "./examples/fastspeech2/conf/fastspeech2.baker.v2.yaml"),
        ("embeddings|encoder", "./examples/fastspeech2/conf/fastspeech2.kss.v1.yaml"),
        ("embeddings|encoder", "./examples/fastspeech2/conf/fastspeech2.kss.v2.yaml"),
    ],
)
def test_fastspeech2_train_some_layers(var_train_expr, config_path):
    config = FastSpeech2Config(n_speakers=5)
    model = TFFastSpeech2(config)
    model._build()
    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config.update({"outdir": "./"})
    config.update({"var_train_expr": var_train_expr})

    STRATEGY = return_strategy()

    trainer = FastSpeech2Trainer(
        config=config, strategy=STRATEGY, steps=0, epochs=0, is_mixed_precision=False,
    )
    trainer.compile(model, optimizer)

    len_trainable_vars = len(trainer._trainable_variables)
    all_trainable_vars = len(model.trainable_variables)

    if var_train_expr is None:
        tf.debugging.assert_equal(len_trainable_vars, all_trainable_vars)
    else:
        tf.debugging.assert_less(len_trainable_vars, all_trainable_vars)


@pytest.mark.parametrize("num_hidden_layers,n_speakers", [(2, 1), (3, 2), (4, 3)])
def test_fastspeech_trainable(num_hidden_layers, n_speakers):
    config = FastSpeech2Config(
        encoder_num_hidden_layers=num_hidden_layers,
        decoder_num_hidden_layers=num_hidden_layers + 1,
        n_speakers=n_speakers,
    )

    fastspeech2 = TFFastSpeech2(config, name="fastspeech")
    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    # fake inputs
    input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
    attention_mask = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
    speaker_ids = tf.convert_to_tensor([0], tf.int32)
    duration_gts = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
    f0_gts = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.float32)
    energy_gts = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.float32)

    mel_gts = tf.random.uniform(shape=[1, 10, 80], dtype=tf.float32)

    @tf.function
    def one_step_training():
        with tf.GradientTape() as tape:
            mel_outputs_before, _, duration_outputs, _, _ = fastspeech2(
                input_ids, speaker_ids, duration_gts, f0_gts, energy_gts, training=True,
            )
            duration_loss = tf.keras.losses.MeanSquaredError()(
                duration_gts, duration_outputs
            )
            mel_loss = tf.keras.losses.MeanSquaredError()(mel_gts, mel_outputs_before)
            loss = duration_loss + mel_loss
        gradients = tape.gradient(loss, fastspeech2.trainable_variables)
        optimizer.apply_gradients(zip(gradients, fastspeech2.trainable_variables))

        tf.print(loss)

    import time

    for i in range(2):
        if i == 1:
            start = time.time()
        one_step_training()
    print(time.time() - start)
