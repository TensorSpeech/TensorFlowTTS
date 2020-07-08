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
        f0_kernel_size,
        energy_kernel_size,
        f0_dropout_rate,
        energy_dropout_rate,
        f0_energy_predictor_filters,
        f0_energy_predictor_kernel_sizes,
        f0_energy_predictor_dropout_probs,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.f0_kernel_size = f0_kernel_size
        self.energy_kernel_size = energy_kernel_size
        self.f0_dropout_rate = f0_dropout_rate
        self.energy_dropout_rate = energy_dropout_rate
        self.f0_energy_predictor_filters = f0_energy_predictor_filters
        self.f0_energy_predictor_kernel_sizes = f0_energy_predictor_kernel_sizes
        self.f0_energy_predictor_dropout_probs = f0_energy_predictor_dropout_probs
