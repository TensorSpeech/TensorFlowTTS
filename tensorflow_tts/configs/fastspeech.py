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

import collections

from tensorflow_tts.processor.ljspeech import LJSPEECH_SYMBOLS as lj_symbols
from tensorflow_tts.processor.kss import KSS_SYMBOLS as kss_symbols
from tensorflow_tts.processor.baker import BAKER_SYMBOLS as bk_symbols
from tensorflow_tts.processor.libritts import LIBRITTS_SYMBOLS as lbri_symbols


SelfAttentionParams = collections.namedtuple(
    "SelfAttentionParams",
    [
        "n_speakers",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "attention_head_size",
        "intermediate_size",
        "intermediate_kernel_size",
        "hidden_act",
        "output_attentions",
        "output_hidden_states",
        "initializer_range",
        "hidden_dropout_prob",
        "attention_probs_dropout_prob",
        "layer_norm_eps",
        "max_position_embeddings",
    ],
)


class FastSpeechConfig(object):
    """Initialize FastSpeech Config."""

    def __init__(
        self,
        dataset='ljspeech',
        vocab_size=len(lj_symbols),
        n_speakers=1,
        encoder_hidden_size=384,
        encoder_num_hidden_layers=4,
        encoder_num_attention_heads=2,
        encoder_attention_head_size=192,
        encoder_intermediate_size=1024,
        encoder_intermediate_kernel_size=3,
        encoder_hidden_act="mish",
        decoder_hidden_size=384,
        decoder_num_hidden_layers=4,
        decoder_num_attention_heads=2,
        decoder_attention_head_size=192,
        decoder_intermediate_size=1024,
        decoder_intermediate_kernel_size=3,
        decoder_hidden_act="mish",
        output_attentions=True,
        output_hidden_states=True,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        max_position_embeddings=2048,
        num_duration_conv_layers=2,
        duration_predictor_filters=256,
        duration_predictor_kernel_sizes=3,
        num_mels=80,
        duration_predictor_dropout_probs=0.1,
        n_conv_postnet=5,
        postnet_conv_filters=512,
        postnet_conv_kernel_sizes=5,
        postnet_dropout_rate=0.1,
        **kwargs
    ):
        """Init parameters for Fastspeech model."""
        # encoder params
        if dataset == "ljspeech":
            self.vocab_size = vocab_size
        elif dataset == "kss":
            self.vocab_size = len(kss_symbols)
        elif dataset == "baker":
            self.vocab_size = len(bk_symbols)
        elif dataset == "libritts":
            self.vocab_size = len(lbri_symbols)
        else:
            raise ValueError("No such dataset: {}".format(dataset))
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.n_speakers = n_speakers
        self.layer_norm_eps = layer_norm_eps

        # encoder params
        self.encoder_self_attention_params = SelfAttentionParams(
            n_speakers=n_speakers,
            hidden_size=encoder_hidden_size,
            num_hidden_layers=encoder_num_hidden_layers,
            num_attention_heads=encoder_num_attention_heads,
            attention_head_size=encoder_attention_head_size,
            hidden_act=encoder_hidden_act,
            intermediate_size=encoder_intermediate_size,
            intermediate_kernel_size=encoder_intermediate_kernel_size,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=max_position_embeddings,
        )

        # decoder params
        self.decoder_self_attention_params = SelfAttentionParams(
            n_speakers=n_speakers,
            hidden_size=decoder_hidden_size,
            num_hidden_layers=decoder_num_hidden_layers,
            num_attention_heads=decoder_num_attention_heads,
            attention_head_size=decoder_attention_head_size,
            hidden_act=decoder_hidden_act,
            intermediate_size=decoder_intermediate_size,
            intermediate_kernel_size=decoder_intermediate_kernel_size,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=max_position_embeddings,
        )

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
