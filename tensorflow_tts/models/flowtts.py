# -*- coding: utf-8 -*-
# Copyright 2020 Mokke Meguru (@MokkeMeguru) Minh Nguyen (@dathudeptrai)
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
"""Flow-TTS model."""
import numpy as np

from typing import Dict
from typing import Tuple

import tensorflow as tf

from tensorflow_tts.configs import FlowTTSConfig

from tensorflow_tts.models.tacotron2 import TFTacotronEncoder
from tensorflow_tts.models.fastspeech2 import TFFastSpeechVariantPredictor
from tensorflow_tts.models.fastspeech import get_initializer
from tensorflow_tts.models.fastspeech import TFFastSpeechSelfAttention

from TFGENZOO.flows.inv1x1conv import Inv1x1Conv2DWithMask
from TFGENZOO.flows.cond_affine_coupling import ConditionalAffineCoupling2DWithMask
from TFGENZOO.flows.squeeze import Squeeze2DWithMask
from TFGENZOO.flows.factor_out import FactorOutWithMask, FactorOut2DWithMask
from TFGENZOO.layers.flowtts_coupling import CouplingBlock
from TFGENZOO.flows.utils.conv_zeros import Conv1DZeros

from TFGENZOO.flows.flowbase import ConditionalFlowModule
from TFGENZOO.flows.flowbase import FactorOutBase
from TFGENZOO.flows.utils import gaussianize
from TFGENZOO.flows.utils.util import split_feature


class TFFlowTTSEncoder(TFTacotronEncoder):
    """Flow-TTS Encoder."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)


class TFFlowTTSLengthPredictor(TFFastSpeechVariantPredictor):
    """Flow-TTS length predictor."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def call(self, inputs, training=False):
        """Call logic."""
        outputs = super().call(inputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = tf.math.reduce_sum(outputs, axis=-1)  # [B]
        return outputs


class TFFlowTTSAttentionPositional(TFFastSpeechSelfAttention):
    """Attention Positional module for flow-tts."""

    def __init__(self, config, **kwargs):
        """Init variables."""
        super().__init__(config, **kwargs)
        self.query = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="query",
        )

    def call(self, inputs, training=False):
        """Call logic."""
        hidden_states, decoder_positional, attention_mask = inputs

        batch_size = tf.shape(hidden_states)[0]
        mixed_key_layer = self.key(hidden_states)  # [B, char_length, F]
        mixed_query_layer = self.query(decoder_positional)  # [B, mel_length, F]
        mixed_value_layer = self.value(hidden_states)  # [B, char_length, F]

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        attention_scores = tf.matmul(
            query_layer, key_layer, transpose_b=True
        )  # [B, n_head, mel_len, char_len]
        dk = tf.cast(tf.shape(key_layer)[-1], tf.float32)  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # extended_attention_masks for self attention encoder.
            extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
            extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
            attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer, (batch_size, -1, self.all_head_size)
        )  # [B, mel_len, F]

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


def build_flow_step(
    step_num: int,
    coupling_depth: int,
    conditional_input: tf.keras.layers.Input,
    scale_type: str = "safe_exp",
):
    """Utility function to construct step-of-flow.

    Sources:

        Flow-TTS's Figure 1

    Args:
        step_num       (int): K in Flow-TTS's Figure 1 (a).
            Number of flow-step iterations
        coupling_depth (int): coupling block's depth
        conditional_input (tf.keras.layers.Input): conditional Input Tensor
        scale_type     (str): Affine Coupling scale function: log_scale -> scale

    Returns:
        ConditionalFlowModule: flow-step's Module

    Examples:

    """

    def CouplingBlockTemplate(x: tf.Tensor):
        cb = CouplingBlock(x, cond=conditional_input, depth=coupling_depth)
        return cb

    cfml = []
    for i in range(step_num):

        # Sources:
        #
        #    FLow-TTS's Figure 1 (b)

        # Inv1x1Conv
        inv1x1 = Inv1x1Conv2DWithMask(name="inv_conv1x1_._{}".format(i))

        # CouplingBlock
        couplingBlockTemplate = CouplingBlockTemplate

        # Affine_xform + Coupling Block
        #
        # Notes:
        #
        #     * forward formula
        #         |
        #         |  where x is source input [B, T, C]
        #         |        c is conditional input [B, T, C'] where C' can be difference with C
        #         |
        #         |  x_1, x_2 = split(x)
        #         |  z_1 = x_1
        #         |  [logs, shift] = NN(x_1, c)
        #         |  z_2 = (x_2 + shift) * exp(logs)
        #    * Coupling Block formula
        #         |
        #         |  where x_1', x_1'' is [B, T, C''] where C'' can be difference with C and C'
        #         |        logs, shift is [B, T, C]
        #         |
        #         |  x_1' =  1x1Conv_1(x_1)
        #         |  x_1'' = GTU(x_1', c)
        #         |  [logs, shift] = 1x1Conv_2(x_1'' + x')
        #         |
        #         |  GTU(x_1, c) = tanh(W_{f, k} * x_1) \odot \sigma(W_{g, k} * c)
        #         |  where W_{f, k} and W_{g, k} are 1-D convolution
        #
        conditionalAffineCoupling = ConditionalAffineCoupling2DWithMask(
            scale_shift_net_template=couplingBlockTemplate, scale_type=scale_type
        )
        cfml.append(inv1x1)
        cfml.append(conditionalAffineCoupling)
    return ConditionalFlowModule(cfml)


class TFFlowTTSDecoder(tf.keras.Model):
    """FlowTTSDecoder module."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        conditionalInput = tf.keras.layers.Input(
            [None, self.config.hidden_size * self.config.n_squeeze]
        )

        squeeze = Squeeze2DWithMask(n_squeeze=self.config.n_squeeze)

        flow_step_1 = build_flow_step(
            step_num=self.config.flow_step_depth,
            coupling_depth=self.config.hidden_size,
            conditional_input=conditionalInput,
            scale_type=self.config.scale_type,
        )

        factor_out_1 = FactorOut2DWithMask(
            with_zaux=False, conditional=self.config.conditional_factor_out
        )

        flow_step_2 = build_flow_step(
            step_num=self.config.flow_step_depth,
            coupling_depth=self.config.hidden_size,
            conditional_input=conditionalInput,
            scale_type=self.config.scale_type,
        )

        factor_out_2 = FactorOut2DWithMask(
            with_zaux=True, conditional=self.config.conditional_factor_out
        )

        flow_step_3 = build_flow_step(
            step_num=self.config.last_flow_step_depth,
            coupling_depth=self.config.hidden_size,
            conditional_input=conditionalInput,
            scale_type=self.config.scale_type,
        )

        self.flows = [
            squeeze,
            flow_step_1,
            factor_out_1,
            flow_step_2,
            factor_out_2,
            flow_step_3,
        ]

    def call(
        self,
        x: tf.Tensor,
        cond: tf.Tensor,
        zaux: tf.Tensor = None,
        mask: tf.Tensor = None,
        inverse: bool = False,
        training: bool = True,
        temparature: float = 1.0,
        **kwargs
    ):
        """Call logic.
        Args:
           x       (tf.Tensor): base input tensor [B, T, C]
           cond    (tf.Tensor): conditional input tensor [B, T, C']
           mask    (tf.Tensor): tensor has sequence length information [B, T]
           inverse      (bool): the flag of the invertible network
           training     (bool): training flag
           temparature (float): sampling temparature

        Notes:
            * forward returns
                - z                (tf.Tensor) [B, T, C_1]
                - log_det_jacobian (tf.Tensor) [B]
                - zaux             (tf.Tensor) [B, T, C_2] where C = C_1 + C_2
                - log_likelihood   (tf.Tensor) [B]
           * inverse returns
                - x                        (tf.Tensor) [B, T, C_1]
                - inverse_log_det_jacobian (tf.Tensor) [B]
        """
        if inverse:
            return self.inverse(
                x,
                cond=cond,
                zaux=zaux,
                mask=mask,
                training=training,
                temparature=temparature,
                **kwargs
            )
        else:
            return self.forward(x, cond=cond, training=training, mask=mask, **kwargs)

    def inverse(
        self,
        x: tf.Tensor,
        cond: tf.Tensor,
        zaux: tf.Tensor,
        training: bool,
        mask: tf.Tensor,
        temparature: float,
        **kwargs
    ):
        """inverse function.
        latent -> object
        """
        inverse_log_det_jacobian = tf.zeros(tf.shape(x)[0:1])

        for flow in reversed(self.flows):
            if isinstance(flow, Squeeze2DWithMask):
                if zaux is not None:
                    x, mask, zaux = flow(x, zaux=zaux, mask=mask, inverse=True,)
                else:
                    x, mask = flow(x, mask=mask, inverse=True,)
            elif isinstance(flow, FactorOutBase):
                if flow.with_zaux:
                    x, zaux = flow(
                        x, zaux=zaux, inverse=True, mask=mask, temparature=temparature
                    )
                else:
                    x = flow(
                        x, zaux=zaux, inverse=True, mask=mask, temparature=temparature
                    )
            else:
                x, ildj = flow(x, cond=cond, inverse=True, training=training, mask=mask)
                inverse_log_det_jacobian += ildj
        return x, inverse_log_det_jacobian

    def forward(
        self, x: tf.Tensor, cond: tf.Tensor, training: bool, mask: tf.Tensor, **kwargs
    ):
        """Forward function.
        object -> latent
        """
        zaux = None
        log_det_jacobian = tf.zeros(tf.shape(x)[0:1])
        log_likelihood = tf.zeros(tf.shape(x)[0:1])
        for flow in self.flows:
            if isinstance(flow, Squeeze2DWithMask):
                if flow.with_zaux:
                    x, mask, zaux = flow(x, zaux=zaux)
                else:
                    x, mask = flow(x)
            elif isinstance(flow, FactorOutBase):
                if x is None:
                    raise Exception()
                x, zaux, ll = flow(x, zaux=zaux, mask=mask)
                log_likelihood += ll
            else:
                x, ldj = flow(x, cond=cond, training=training, mask=mask)
                log_det_jacobian += ldj
        return x, log_det_jacobian, zaux, log_likelihood, mask


class TFFlowTTS(tf.keras.Model):
    """Flow-TTS modules."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.encoder = TFFlowTTSEncoder(config=config, name="encoder")
        self.length_predictor = TFFlowTTSLengthPredictor(
            config, name="length_predictor"
        )
        self.attention_positional = TFFlowTTSAttentionPositional(
            config, name="attention_positional"
        )
        self.decoder = TFFlowTTSDecoder(config=config, name="decoder")

        self.config = config

    def _squeeze_feats(self, x, n_squeeze):
        _, t, c = x.shape

        t = (t // n_squeeze) * n_squeeze
        x = x[:, :t, :]  # [B, t_round, c]

        x = tf.reshape(
            tf.reshape(x, [-1, t // n_squeeze, n_squeeze, c]),
            [-1, t // n_squeeze, c * n_squeeze],
        )
        return x

    def call(
        self,
        input_ids,
        attention_mask,
        speaker_ids,
        mel_gts,
        mel_lengths,
        training=False,
    ):
        """Call logic."""
        # Encoder Step.
        encoder_hidden_states = self.encoder(
            [input_ids, speaker_ids, attention_mask], training=training
        )

        # predict mel lengths
        mel_length_predictions = self.length_predictor(
            [encoder_hidden_states, attention_mask], training=training
        )

        # calculate conditional feature for flow.
        decoder_pos = tf.range(1, tf.shape(mel_gts)[1] + 1, dtype=tf.int32)
        mask = tf.sequence_mask(
            mel_lengths, maxlen=tf.reduce_max(mel_lengths), dtype=decoder_pos.dtype
        )
        masked_decoder_pos = tf.expand_dims(decoder_pos, 0) * mask
        conditional_feats = self.attention_positional(
            [encoder_hidden_states, masked_decoder_pos, attention_mask],
            training=training,
        )[0]

        # mask and squeeze conditional feature.
        conditional_feats *= tf.cast(
            tf.expand_dims(mask, -1), dtype=conditional_feats.dtype
        )
        squeeze_condition_feats = self._squeeze_feats(
            conditional_feats, self.config.n_squeeze
        )

        z, log_det_jacobian, zaux, log_likelihood, mask = self.decoder(
            mel_gts, cond=squeeze_condition_feats, mask=mask[:, :, None], inverse=False
        )

        # here is debug
        tf.print("-----------------")
        tf.print("reconstruction")
        rev_x, ildj = self.decoder(
            z, zaux=zaux, cond=squeeze_condition_feats, mask=mask, inverse=True
        )
        tf.print(rev_x.shape, mel_gts.shape)
        tf.print("reconstruction diff", tf.reduce_mean(rev_x - mel_gts))
        tf.print("ildj + ldj", tf.reduce_sum(log_det_jacobian + ildj))

        tf.print("-----------------")
        tf.print("conditional generation")
        rev_x, ildj = self.decoder(
            z, cond=squeeze_condition_feats, mask=mask, inverse=True
        )
        tf.print(rev_x.shape, mel_gts.shape)
        tf.print("reconstruction diff", tf.reduce_mean(rev_x - mel_gts))

        tf.print("-----------------")
        tf.print("generation")
        z = tf.random.normal(z.shape)
        rev_x, ildj = self.decoder(
            z, cond=squeeze_condition_feats, mask=mask, inverse=True
        )
        tf.print(rev_x.shape, mel_gts.shape)
        tf.print("-----------------")
        return z, log_det_jacobian, zaux, log_likelihood, mel_length_predictions, mask

    def inference(self):
        # TODO (@MokkeMeguru)
        return


if __name__ == "__main__":
    flowtts_config = FlowTTSConfig()
    flowtts = TFFlowTTS(config=flowtts_config, name="flow_tts")

    mel_gts = tf.random.normal([1, 32, 64])

    # forward steps.
    (
        z,
        log_det_jacobian,
        zaux,
        log_likelihood,
        mel_length_predictions,
        output_mask,
    ) = flowtts(
        input_ids=tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32),
        attention_mask=tf.constant([[1, 1, 1, 1, 1]], dtype=tf.int32),
        speaker_ids=tf.constant([0], dtype=tf.int32),
        mel_gts=mel_gts,
        mel_lengths=tf.constant([mel_gts.shape[1]], dtype=tf.int32),
    )

    flowtts.summary()
