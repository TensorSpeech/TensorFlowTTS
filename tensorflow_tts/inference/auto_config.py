# -*- coding: utf-8 -*-
# Copyright 2020 The HuggingFace Inc. team and Minh Nguyen (@dathudeptrai)
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
"""Tensorflow Auto Config modules."""

import logging
import yaml
from collections import OrderedDict

from tensorflow_tts.configs import (
    FastSpeechConfig,
    FastSpeech2Config,
    MelGANGeneratorConfig,
    MultiBandMelGANGeneratorConfig,
    HifiGANGeneratorConfig,
    Tacotron2Config,
    ParallelWaveGANGeneratorConfig,
)

CONFIG_MAPPING = OrderedDict(
    [
        ("fastspeech", FastSpeechConfig),
        ("fastspeech2", FastSpeech2Config),
        ("multiband_melgan_generator", MultiBandMelGANGeneratorConfig),
        ("melgan_generator", MelGANGeneratorConfig),
        ("hifigan_generator", HifiGANGeneratorConfig),
        ("tacotron2", Tacotron2Config),
        ("parallel_wavegan_generator", ParallelWaveGANGeneratorConfig)
    ]
)


class AutoConfig:
    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        with open(pretrained_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)

        try:
            model_type = config["model_type"]
            config_class = CONFIG_MAPPING[model_type]
            config_class = config_class(**config[model_type + "_params"], **kwargs)
            return config_class
        except Exception:
            raise ValueError(
                "Unrecognized config in {}. "
                "Should have a `model_type` key in its config.yaml, or contain one of the following strings "
                "in its name: {}".format(
                    pretrained_path, ", ".join(CONFIG_MAPPING.keys())
                )
            )
