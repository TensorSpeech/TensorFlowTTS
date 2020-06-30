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
"""FastSpeech Config object."""

from tensorflow_tts.processor.ljspeech import symbols


class FastSpeechConfig(object):
    """Initialize FastSpeech Config."""

    def __init__(
        self,
        vocab_size=len(symbols),
        n_speakers=1,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=2,
        intermediate_size=1536,
        intermediate_kernel_size=3,
        num_duration_conv_layers=2,
        duration_predictor_filters=256,
        duration_predictor_kernel_sizes=3,
        num_mels=80,
        hidden_act="mish",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        duration_predictor_dropout_probs=0.1,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        output_attentions=True,
        output_hidden_states=True,
        n_conv_postnet=5,
        postnet_conv_filters=512,
        postnet_conv_kernel_sizes=5,
        postnet_dropout_rate=0.1,
        **kwargs
    ):
        """Init parameters for Fastspeech model."""
        # fastspeech
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.intermediate_kernel_size = intermediate_kernel_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.n_speakers = n_speakers
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.duration_predictor_dropout_probs = duration_predictor_dropout_probs
        self.num_duration_conv_layers = num_duration_conv_layers
        self.duration_predictor_filters = duration_predictor_filters
        self.duration_predictor_kernel_sizes = duration_predictor_kernel_sizes
        self.num_mels = num_mels

        # postnet
        self.n_conv_postnet = n_conv_postnet
        self.postnet_conv_filters = postnet_conv_filters
        self.postnet_conv_kernel_sizes = postnet_conv_kernel_sizes
        self.postnet_dropout_rate = postnet_dropout_rate
