# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen Quan Anh (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)

"""FastSpeech Config object."""

from tensorflow_tts.processor.ljspeech import symbols


class AlignTTSConfig(object):
    """Initialize FastSpeech Config."""

    def __init__(
            self,
            vocab_size=len(symbols),
            f0_energy_quantize_size=256,
            n_speakers=1,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=2,
            intermediate_size=1536,
            intermediate_kernel_size=3,
            num_duration_conv_layers=2,
            num_f0_energy_conv_layers=2,
            duration_predictor_filters=256,
            duration_predictor_kernel_sizes=3,
            f0_energy_predictor_filters=384,
            f0_energy_predictor_kernel_sizes=3,
            f0_energy_min_threshold=0.0,
            f0_energy_max_threshold=4.0,
            num_mels=80,
            hidden_act="mish",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            duration_predictor_dropout_probs=0.1,
            f0_energy_dropout_probs=0.5,
            max_position_embeddings=2048,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            output_attentions=True,
            output_hidden_states=True,
            n_conv_postnet=5,
            postnet_conv_filters=512,
            postnet_conv_kernel_sizes=5,
            postnet_dropout_rate=0.1,
            num_mdn_dense_layers=2,
            mdn_hidden_size=256,
            mdn_dropout_prob=0.5):
        """Init parameters for Fastspeech model."""
        # fastspeech
        self.vocab_size = vocab_size
        self.f0_energy_quantize_size = f0_energy_quantize_size
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
        self.f0_energy_dropout_probs = f0_energy_dropout_probs
        self.num_duration_conv_layers = num_duration_conv_layers
        self.num_f0_energy_conv_layers = num_f0_energy_conv_layers
        self.duration_predictor_filters = duration_predictor_filters
        self.duration_predictor_kernel_sizes = duration_predictor_kernel_sizes
        self.f0_energy_predictor_filters = f0_energy_predictor_filters
        self.f0_energy_predictor_kernel_sizes = f0_energy_predictor_kernel_sizes
        self.num_mels = num_mels
        self.f0_energy_min_threshold = f0_energy_min_threshold
        self.f0_energy_max_threshold = f0_energy_max_threshold
        self.num_mdn_dense_layers = num_mdn_dense_layers
        self.mdn_hidden_size = mdn_hidden_size
        self.mdn_dropout_prob = mdn_dropout_prob

        # postnet
        self.n_conv_postnet = n_conv_postnet
        self.postnet_conv_filters = postnet_conv_filters
        self.postnet_conv_kernel_sizes = postnet_conv_kernel_sizes
        self.postnet_dropout_rate = postnet_dropout_rate
