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

import pytest
import tensorflow as tf

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import AutoProcessor
from tensorflow_tts.inference import TFAutoModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


@pytest.mark.parametrize(
    "mapper_path", 
    [
        "./test/files/baker_mapper.json",
        "./test/files/kss_mapper.json",
        "./test/files/libritts_mapper.json",
        "./test/files/ljspeech_mapper.json",
     ]
)
def test_auto_processor(mapper_path):
    processor = AutoProcessor.from_pretrained(pretrained_path=mapper_path)


@pytest.mark.parametrize(
    "config_path", 
    [
        "./examples/fastspeech/conf/fastspeech.v1.yaml", 
        "./examples/fastspeech/conf/fastspeech.v3.yaml", 
        "./examples/fastspeech2/conf/fastspeech2.v1.yaml",
        "./examples/fastspeech2/conf/fastspeech2.v2.yaml",
        "./examples/fastspeech2/conf/fastspeech2.kss.v1.yaml",
        "./examples/fastspeech2/conf/fastspeech2.kss.v2.yaml",
        "./examples/melgan/conf/melgan.v1.yaml",
        "./examples/melgan_stft/conf/melgan_stft.v1.yaml",
        "./examples/multiband_melgan/conf/multiband_melgan.v1.yaml",
        "./examples/tacotron2/conf/tacotron2.v1.yaml",
        "./examples/tacotron2/conf/tacotron2.kss.v1.yaml",
        "./examples/parallel_wavegan/conf/parallel_wavegan.v1.yaml",
        "./examples/hifigan/conf/hifigan.v1.yaml",
        "./examples/hifigan/conf/hifigan.v2.yaml",
     ]
)
def test_auto_model(config_path):
    config = AutoConfig.from_pretrained(pretrained_path=config_path)
    model = TFAutoModel.from_pretrained(config=config, pretrained_path=None)
