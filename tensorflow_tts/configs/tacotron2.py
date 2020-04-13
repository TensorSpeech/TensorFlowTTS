# -*- coding: utf-8 -*-

# Copyright 2020 Minh Nguyen Quan Anh (@dathudeptrai)
#  MIT License (https://opensource.org/licenses/MIT)
"""Tacotron-2 Config object."""


class Tacotron2Config(object):
    """Initialize Tacotron-2 Config."""

    def __init__(
            self,
            vocab_size=250,
            embedding_hidden_size=512,
            initializer_range=0.02,
            layer_norm_eps=1e-6,
            embedding_dropout_prob=0.1,
            n_speakers=5,
            n_conv_encoder=3,
            encoder_conv_filters=512,
            encoder_conv_kernel_sizes=5,
            encoder_activation='relu',
            encoder_lstm_units=256,
            n_prenet_layers=2,
            prenet_units=256,
            prenet_activation='relu',
            prenet_dropout_rate=0.5,
            n_lstm_decoder=2,
            decoder_lstm_units=1024,
            attention_dim=128,
            memory_units=256,  # not neccessary
            attention_filters=32,
            attention_kernel=31,
            n_mels=80,
            n_conv_postnet=5,
            postnet_conv_filters=512,
            postnet_conv_kernel_sizes=5,
            postnet_dropout_rate=0.1):
        """Init parameters for Tacotron-2 model."""
        self.vocab_size = vocab_size
        self.embedding_hidden_size = embedding_hidden_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embedding_dropout_prob = embedding_dropout_prob
        self.n_speakers = n_speakers
        self.n_conv_encoder = n_conv_encoder
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_sizes = encoder_conv_kernel_sizes
        self.encoder_activation = encoder_activation
        self.encoder_lstm_units = encoder_lstm_units

        # decoder param
        self.n_prenet_layers = n_prenet_layers
        self.prenet_units = prenet_units
        self.prenet_activation = prenet_activation
        self.prenet_dropout_rate = prenet_dropout_rate
        self.n_lstm_decoder = n_lstm_decoder
        self.decoder_lstm_units = decoder_lstm_units
        self.attention_dim = attention_dim
        self.memory_units = memory_units
        self.attention_filters = attention_filters
        self.attention_kernel = attention_kernel
        self.n_mels = n_mels

        # postnet
        self.n_conv_postnet = n_conv_postnet
        self.postnet_conv_filters = postnet_conv_filters
        self.postnet_conv_kernel_sizes = postnet_conv_kernel_sizes
        self.postnet_dropout_rate = postnet_dropout_rate
