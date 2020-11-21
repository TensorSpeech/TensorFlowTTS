"""Gradient Accummlate for training TF2 custom training loop.
Copy from https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py.
"""


import re

import tensorflow as tf


class GradientAccumulator(object):
    """Gradient accumulation utility.
    When used with a distribution strategy, the accumulator should be called in a
    replica context. Gradients will be accumulated locally on each replica and
    without synchronization. Users should then call ``.gradients``, scale the
    gradients if required, and pass the result to ``apply_gradients``.
    """

    # We use the ON_READ synchronization policy so that no synchronization is
    # performed on assignment. To get the value, we call .value() which returns the
    # value on the current replica without synchronization.

    def __init__(self):
        """Initializes the accumulator."""
        self._gradients = []
        self._accum_steps = None

    @property
    def step(self):
        """Number of accumulated steps."""
        if self._accum_steps is None:
            self._accum_steps = tf.Variable(
                tf.constant(0, dtype=tf.int64),
                trainable=False,
                synchronization=tf.VariableSynchronization.ON_READ,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )

        return self._accum_steps.value()

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        if not self._gradients:
            raise ValueError(
                "The accumulator should be called first to initialize the gradients"
            )
        return list(
            gradient.value() if gradient is not None else gradient
            for gradient in self._gradients
        )

    def __call__(self, gradients):
        """Accumulates :obj:`gradients` on the current replica."""
        if not self._gradients:
            _ = self.step  # Create the step variable.
            self._gradients.extend(
                [
                    tf.Variable(
                        tf.zeros_like(gradient),
                        trainable=False,
                        synchronization=tf.VariableSynchronization.ON_READ,
                    )
                    if gradient is not None
                    else gradient
                    for gradient in gradients
                ]
            )
        if len(gradients) != len(self._gradients):
            raise ValueError(
                "Expected %s gradients, but got %d"
                % (len(self._gradients), len(gradients))
            )

        for accum_gradient, gradient in zip(self._gradients, gradients):
            if accum_gradient is not None and gradient is not None:
                accum_gradient.assign_add(gradient, read_value=False)

        self._accum_steps.assign_add(1)

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        if not self._gradients:
            return
        self._accum_steps.assign(0)
        for gradient in self._gradients:
            if gradient is not None:
                gradient.assign(tf.zeros_like(gradient), read_value=False)
