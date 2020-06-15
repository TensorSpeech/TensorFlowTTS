#!/usr/bin/env python3

from typing import Dict
import tensorflow as tf

from TFGENZOO.flows import FactorOutBase
from tensorflow_tts.flows.utils.flow_step import build_flow_step
from tensorflow_tts.flows.utils.factor_out import FactorOutWithMask
from utils.squeeze2D import Squeeze2D


class FlowTTSDecoder(tf.keras.Model):
    def __init__(self, hparams: Dict, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.build_model()

    def build_model(self):
        conditionalInput = tf.keras.layers.Input(
            [None, self.hparams["conditional_width"]]
        )

        squeeze = Squeeze2D()

        flow_step_1 = build_flow_step(
            step_num=self.hparams["flow_step_depth"],
            coupling_depth=self.hparams["conditional_depth"],
            conditional_input=conditionalInput,
            scale_type=self.hparams["scale_type"],
        )

        factor_out_1 = FactorOutWithMask(
            with_zaux=False, conditional=self.hparams["conditional_factor_out"]
        )

        flow_step_2 = build_flow_step(
            step_num=self.hparams["flow_step_depth"],
            coupling_depth=self.hparams["conditional_depth"],
            conditional_input=conditionalInput,
            scale_type=self.hparams["scale_type"],
        )

        factor_out_2 = FactorOutWithMask(
            with_zaux=True, conditional=self.hparams["conditional_factor_out"]
        )

        flow_step_3 = build_flow_step(
            step_num=self.hparams["last_flow_step_depth"],
            coupling_depth=self.hparams["conditional_depth"],
            conditional_input=conditionalInput,
            scale_type=self.hparams["scale_type"],
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
        """
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
        """inverse function
        latent -> object
        """
        inverse_log_det_jacobian = tf.zeros(tf.shape(x)[0:1])

        for flow in reversed(self.flows):
            if isinstance(flow, Squeeze2D):
                x, zaux = flow(x, zaux=zaux, mask=mask, inverse=True)
                if mask is not None:
                    _, t = mask.shape
                    mask = tf.reshape(
                        tf.tile(tf.expand_dims(mask, -1), [1, 1, 2]), [-1, t * 2]
                    )

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
        """forward function
        object -> latent
        """
        zaux = None
        log_det_jacobian = tf.zeros(tf.shape(x)[0:1])
        log_likelihood = tf.zeros(tf.shape(x)[0:1])
        for flow in self.flows:
            if isinstance(flow, Squeeze2D):
                if flow.with_zaux:
                    x, zaux = flow(x, zaux=zaux)
                else:
                    x = flow(x)
                _, t = mask.shape
                mask = tf.reshape(mask, [-1, t // 2, 2])[..., 0]
            elif isinstance(flow, FactorOutBase):
                if x is None:
                    raise Exception()
                x, zaux, ll = flow(x, zaux=zaux, mask=mask)
                log_likelihood += ll
            else:
                x, ldj = flow(x, cond=cond, training=training, mask=mask)
                log_det_jacobian += ldj
        return x, log_det_jacobian, zaux, log_likelihood


if __name__ == "__main__":
    hparams = {
        "conditional_width": 128,  # conditional inputs's C' in [B, T, C']
        "flow_step_depth": 4,
        "last_flow_step_depth": 2,
        "conditional_depth": 256,
        "scale_type": "exp",
        "conditional_factor_out": True,
    }
    model = FlowTTSDecoder(hparams)
    x = tf.random.normal([64, 32, 64])
    cond = tf.random.normal([64, 16, 128])
    mask = tf.sequence_mask(tf.ones([64]) * 30, maxlen=32)
    model(x, cond=cond, mask=mask, inverse=False)
    model.summary()

    logdir = "logs"
    writer = tf.summary.create_file_writer(logdir)

    @tf.function
    def my_func(x, cond, mask):
        return model(x, cond=cond, mask=mask, inverse=False)

    tf.summary.trace_on(graph=True)
    results = my_func(x, cond, mask)
    with writer.as_default():
        tf.summary.trace_export(name="flow-tts", step=0)
    tf.summary.trace_off()
