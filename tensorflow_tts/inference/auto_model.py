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
"""Tensorflow Auto Model modules."""

import logging
import warnings
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

from tensorflow_tts.models import (
    TFMelGANGenerator,
    TFMBMelGANGenerator,
    TFHifiGANGenerator,
    TFParallelWaveGANGenerator,
)

from tensorflow_tts.inference.savable_models import (
    SavableTFFastSpeech,
    SavableTFFastSpeech2,
    SavableTFTacotron2
)


TF_MODEL_MAPPING = OrderedDict(
    [
        (FastSpeech2Config, SavableTFFastSpeech2),
        (FastSpeechConfig, SavableTFFastSpeech),
        (MultiBandMelGANGeneratorConfig, TFMBMelGANGenerator),
        (MelGANGeneratorConfig, TFMelGANGenerator),
        (Tacotron2Config, SavableTFTacotron2),
        (HifiGANGeneratorConfig, TFHifiGANGenerator),
        (ParallelWaveGANGeneratorConfig, TFParallelWaveGANGenerator),
    ]
)


class TFAutoModel(object):
    """General model class for inferencing."""

    def __init__(self):
        raise EnvironmentError("Cannot be instantiated using `__init__()`")

    @classmethod
    def from_pretrained(cls, config, pretrained_path=None, **kwargs):
        is_build = kwargs.pop("is_build", True)
        for config_class, model_class in TF_MODEL_MAPPING.items():
            if isinstance(config, config_class) and str(config_class.__name__) in str(
                config
            ):
                model = model_class(config=config, **kwargs)
                if is_build:
                    model._build()
                if pretrained_path is not None and ".h5" in pretrained_path:
                    try:
                        model.load_weights(pretrained_path)
                    except:
                        model.load_weights(
                            pretrained_path, by_name=True, skip_mismatch=True
                        )
                return model
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in TF_MODEL_MAPPING.keys()),
            )
        )
