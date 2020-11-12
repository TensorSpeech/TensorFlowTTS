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
"""FastSpeech2 Config object."""

from tensorflow_tts.configs import FastSpeechConfig


class FastSpeech2Config(FastSpeechConfig):
    """Initialize FastSpeech2 Config."""

    def __init__(
        self,
        variant_prediction_num_conv_layers=2,
        variant_kernel_size=9,
        variant_dropout_rate=0.5,
        variant_predictor_filter=256,
        variant_predictor_kernel_size=3,
        variant_predictor_dropout_rate=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.variant_prediction_num_conv_layers = variant_prediction_num_conv_layers
        self.variant_predictor_kernel_size = variant_predictor_kernel_size
        self.variant_predictor_dropout_rate = variant_predictor_dropout_rate
        self.variant_predictor_filter = variant_predictor_filter
