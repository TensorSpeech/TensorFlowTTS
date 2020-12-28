# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team
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
"""Tensorflow Savable Model modules."""

import numpy as np
import tensorflow as tf

from tensorflow_tts.models import (
    TFFastSpeech,
    TFFastSpeech2,
    TFMelGANGenerator,
    TFMBMelGANGenerator,
    TFHifiGANGenerator,
    TFTacotron2,
    TFParallelWaveGANGenerator,
)


class SavableTFTacotron2(TFTacotron2):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def call(self, inputs, training=False):
        input_ids, input_lengths, speaker_ids = inputs
        return super().inference(input_ids, input_lengths, speaker_ids)

    def _build(self):
        input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=tf.int32)
        input_lengths = tf.convert_to_tensor([9], dtype=tf.int32)
        speaker_ids = tf.convert_to_tensor([0], dtype=tf.int32)
        self([input_ids, input_lengths, speaker_ids])


class SavableTFFastSpeech(TFFastSpeech):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def call(self, inputs, training=False):
        input_ids, speaker_ids, speed_ratios = inputs
        return super()._inference(input_ids, speaker_ids, speed_ratios)

    def _build(self):
        input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        speed_ratios = tf.convert_to_tensor([1.0], tf.float32)
        self([input_ids, speaker_ids, speed_ratios])


class SavableTFFastSpeech2(TFFastSpeech2):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def call(self, inputs, training=False):
        input_ids, speaker_ids, speed_ratios, f0_ratios, energy_ratios = inputs
        return super()._inference(
            input_ids, speaker_ids, speed_ratios, f0_ratios, energy_ratios
        )

    def _build(self):
        input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
        speaker_ids = tf.convert_to_tensor([0], tf.int32)
        speed_ratios = tf.convert_to_tensor([1.0], tf.float32)
        f0_ratios = tf.convert_to_tensor([1.0], tf.float32)
        energy_ratios = tf.convert_to_tensor([1.0], tf.float32)
        self([input_ids, speaker_ids, speed_ratios, f0_ratios, energy_ratios])
