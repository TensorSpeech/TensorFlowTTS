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
"""Multi-band MelGAN Config object."""

from tensorflow_tts.configs import MelGANDiscriminatorConfig, MelGANGeneratorConfig


class MultiBandMelGANGeneratorConfig(MelGANGeneratorConfig):
    """Initialize Multi-band MelGAN Generator Config."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subbands = kwargs.pop("subbands", 4)
        self.taps = kwargs.pop("taps", 62)
        self.cutoff_ratio = kwargs.pop("cutoff_ratio", 0.142)
        self.beta = kwargs.pop("beta", 9.0)


class MultiBandMelGANDiscriminatorConfig(MelGANDiscriminatorConfig):
    """Initialize Multi-band MelGAN Discriminator Config."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
