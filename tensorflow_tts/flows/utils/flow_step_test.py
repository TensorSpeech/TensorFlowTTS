#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow_tts.flows.utils.flow_step import build_flow_step, _Model


class FlowStepTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.model = _Model()
        x = tf.random.normal([128, 32, 64])
        cond = tf.random.normal([128, 32, 128])
        _ = self.model(x, cond, inverse=False)

    def testFlowStepOutputShape(self):
        x = tf.random.normal([128, 32, 64])
        cond = tf.random.normal([128, 32, 128])
        z, ldj = self.model(x, cond, inverse=False)
        self.assertShapeEqual(np.zeros(x.shape), z)
        self.assertShapeEqual(np.zeros(x.shape[0:1]), ldj)

    def testFlowStepOutput(self):
        x = tf.random.normal([128, 32, 64])
        cond = tf.random.normal([128, 32, 128])
        z, ldj = self.model(x, cond, inverse=False)
        rev_x, ildj = self.model(z, cond, inverse=True)
        self.assertAllClose(x, rev_x, rtol=1e-1)
        self.assertAllClose(ldj + ildj, np.zeros(ldj.shape))
