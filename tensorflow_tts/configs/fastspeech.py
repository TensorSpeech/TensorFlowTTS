# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen Quan Anh (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)
"""FastSpeech Config object."""


class FastSpeechConfig(object):
    """Initialize FastSpeech Config."""

    def __init__(
            self,
            vocab_size=11,
            n_speakers=5,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            num_duration_conv_layers=4,
            duration_predictor_filters=128,
            duration_predictor_kernel_sizes=3,
            num_mels=80,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            duration_predictor_dropout_probs=0.1,
            max_position_embeddings=11,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            output_attentions=False,
            output_hidden_states=False,):
        """Init parameters for Fastspeech model."""
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
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
