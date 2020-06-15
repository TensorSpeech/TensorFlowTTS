#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from tensorflow_tts.flows.utils.coupling_block import CouplingBlock, GTU


class GTUTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.gtu = GTU()
        self.gtu.build((None, None, 16))

    def testGTUOutputShape(self):
        x = tf.random.normal([1024, 12, 16])
        c = tf.random.normal([1024, 12, 128])
        z = self.gtu(x, c=c)
        self.assertShapeEqual(np.zeros(x.shape), z)


class CouplingBlockTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        x = tf.keras.layers.Input([None, 16])
        c = tf.keras.layers.Input([None, 128])
        self.couplingBlock = CouplingBlock(x=x, cond=c, depth=256)

    def testCouplingBlockOutputShape(self):
        x = tf.random.normal([128, 16, 16])
        c = tf.random.normal([128, 16, 128])
        z = self.couplingBlock([x, c], training=True)
        output_shape = list(x.shape)
        output_shape[-1] = output_shape[-1] * 2
        self.assertShapeEqual(np.zeros(output_shape), z)

    def testCouplingBlockFirstOutput(self):
        x = tf.random.normal([128, 32, 16])
        c = tf.random.normal([128, 32, 128])
        z = self.couplingBlock([x, c], training=True)
        output_shape = list(x.shape)
        output_shape[-1] = output_shape[-1] * 2
        self.assertAllEqual(np.zeros(output_shape), z)
