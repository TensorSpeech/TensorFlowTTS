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

    def __init__(self,
                 max_f0_embeddings,
                 max_energy_embeddings,
                 f0_energy_predictor_filters,
                 f0_energy_predictor_kernel_sizes,
                 f0_energy_predictor_dropout_probs,
                 ** kwargs):
        super().__init__(**kwargs)
        self.max_f0_embeddings = max_f0_embeddings
        self.max_energy_embeddings = max_energy_embeddings
        self.f0_energy_predictor_filters = f0_energy_predictor_filters
        self.f0_energy_predictor_kernel_sizes = f0_energy_predictor_kernel_sizes
        self.f0_energy_predictor_dropout_probs = f0_energy_predictor_dropout_probs
