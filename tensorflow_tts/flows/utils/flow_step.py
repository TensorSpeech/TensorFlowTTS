#!/usr/bin/env python3

import tensorflow as tf
from tensorflow_tts.flows.utils.inv1x1conv2D import Inv1x1Conv2DWithMask
from tensorflow_tts.flows.utils.coupling_block import CouplingBlock
from TFGENZOO.flows.flowbase import ConditionalFlowModule
from tensorflow_tts.flows.utils.cond_affine_coupling import (
    ConditionalAffineCouplingWithMask,
)


def build_flow_step(
    step_num: int,
    coupling_depth: int,
    conditional_input: tf.keras.layers.Input,
    scale_type: str = "safe_exp",
):
    """utility function to construct step-of-flow

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
        inv1x1 = Inv1x1Conv2DWithMask()

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
        conditionalAffineCoupling = ConditionalAffineCouplingWithMask(
            scale_shift_net_template=couplingBlockTemplate, scale_type=scale_type
        )
        cfml.append(inv1x1)
        cfml.append(conditionalAffineCoupling)
    return ConditionalFlowModule(cfml)


class _Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        cond = tf.keras.Input([32, 128], name="conditional_input")
        self.flow_step = build_flow_step(
            step_num=4, coupling_depth=256, conditional_input=cond
        )

    def call(self, x, c, inverse=False):
        return self.flow_step(x, cond=c, inverse=inverse)


def logging_test():
    logdir = "logs"
    writer = tf.summary.create_file_writer(logdir)

    model = _Model()
    x = tf.random.normal([128, 32, 64])

    cond = tf.random.normal([128, 32, 128])
    z, ldj = model(x, cond, inverse=False)

    dense = tf.keras.layers.Dense(128)

    @tf.function
    def my_func(x, cond):
        y = model(x, cond, inverse=False)
        return y

    tf.summary.trace_on(graph=True)
    y = my_func(x, cond)
    with writer.as_default():
        tf.summary.trace_export(name="flow_step", step=0, profiler_outdir=logdir)
    tf.summary.trace_off()
    print(z.shape)
    print(ldj.shape)
    rev_x, ildj = model(z, cond)
    print(tf.reduce_mean(x - rev_x))
    print(tf.reduce_mean(ldj + ildj))
    model.summary()
    import pprint

    print("trainable varialbes")
    pprint.pprint([f.name for f in model.variables if f.trainable])
    print("non-trainable variables")
    pprint.pprint([f.name for f in model.variables if not f.trainable])


if __name__ == "__main__":
    logging_test()
