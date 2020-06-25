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
"""Flow-TTS Config object."""

from tensorflow_tts.processor.ljspeech import symbols


class FlowTTSConfig(object):
    """Initialize FlowTTS Config."""

    def __init__(
            self,
            vocab_size=len(symbols),
            embedding_hidden_size=512,
            initializer_range=0.02,
            layer_norm_eps=1e-6,
            embedding_dropout_prob=0.1,
            n_speakers=5,
            n_conv_encoder=3,
            encoder_conv_filters=512,
            encoder_conv_kernel_sizes=5,
            encoder_conv_activation='mish',
            encoder_conv_dropout_rate=0.5,
            encoder_lstm_units=256,
            f0_energy_predictor_filters=256,
            f0_energy_predictor_kernel_sizes=3,
            f0_energy_predictor_dropout_probs=0.5,
            num_duration_conv_layers=2,
            max_position_embeddings=2048,
            output_attentions=False,
            output_hidden_states=False,
            num_attention_heads=2,
            hidden_size=384,
            attention_probs_dropout_prob=0.1,
            n_squeeze=8,
            flow_step_depth=4,
            last_flow_step_depth=2,
            scale_type="exp",
            conditional_factor_out=True,
            **kwargs):
        # encoder parameters
        self.vocab_size = vocab_size
        self.embedding_hidden_size = embedding_hidden_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embedding_dropout_prob = embedding_dropout_prob
        self.n_speakers = n_speakers
        self.n_conv_encoder = n_conv_encoder
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_sizes = encoder_conv_kernel_sizes
        self.encoder_conv_activation = encoder_conv_activation
        self.encoder_conv_dropout_rate = encoder_conv_dropout_rate
        self.encoder_lstm_units = encoder_lstm_units

        # length predictor parameters.
        self.num_duration_conv_layers = num_duration_conv_layers
        self.f0_energy_predictor_filters = f0_energy_predictor_filters
        self.f0_energy_predictor_kernel_sizes = f0_energy_predictor_kernel_sizes
        self.f0_energy_predictor_dropout_probs = f0_energy_predictor_dropout_probs

        # attention positional parameters.
        self.max_position_embeddings = max_position_embeddings
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        # decoder parameters.
        self.n_squeeze = n_squeeze
        self.flow_step_depth = flow_step_depth
        self.last_flow_step_depth = last_flow_step_depth
        self.scale_type = scale_type
        self.conditional_factor_out = conditional_factor_out
