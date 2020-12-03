# -*- coding: utf-8 -*-
# Copyright 2020 The TensorFlowTTS Team.
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
"""Tensorflow Auto Processor modules."""

import logging
import json
from collections import OrderedDict

from tensorflow_tts.processor import (
    LJSpeechProcessor,
    KSSProcessor,
    BakerProcessor,
    LibriTTSProcessor,
    ThorstenProcessor,
)

CONFIG_MAPPING = OrderedDict(
    [
        ("LJSpeechProcessor", LJSpeechProcessor),
        ("KSSProcessor", KSSProcessor),
        ("BakerProcessor", BakerProcessor),
        ("LibriTTSProcessor", LibriTTSProcessor),
        ("ThorstenProcessor", ThorstenProcessor)
    ]
)


class AutoProcessor:
    def __init__(self):
        raise EnvironmentError(
            "AutoProcessor is designed to be instantiated "
            "using the `AutoProcessor.from_pretrained(pretrained_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        with open(pretrained_path, "r") as f:
            config = json.load(f)

        try:
            processor_name = config["processor_name"]
            processor_class = CONFIG_MAPPING[processor_name]
            processor_class = processor_class(
                data_dir=None, loaded_mapper_path=pretrained_path
            )
            return processor_class
        except Exception:
            raise ValueError(
                "Unrecognized processor in {}. "
                "Should have a `processor_name` key in its config.json, or contain one of the following strings "
                "in its name: {}".format(
                    pretrained_path, ", ".join(CONFIG_MAPPING.keys())
                )
            )
