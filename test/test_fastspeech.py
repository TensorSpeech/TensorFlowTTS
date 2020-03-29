import logging
import os
import pytest
import numpy as np
import tensorflow as tf

from tensorflow_tts.layers.fastspeech import TacotronEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")


@pytest.mark.parametrize(
    "hidden_dim,num_conv_layers,conv_kernel_size,conv_dilation_rate,out_dim,use_bias,dropout",
    [(512, 4, 3, 1, 512, True, 0.0), (512, 4, 3, 1, 512, True, 0.1),
     (512, 4, 3, 1, 256, True, 0.1), (512, 4, 3, 3, 256, True, 0.1)])
def test_tacontron_encoder(hidden_dim,
                           num_conv_layers,
                           conv_kernel_size,
                           conv_dilation_rate,
                           out_dim,
                           use_bias,
                           dropout):
    fake_input_1d = tf.random.normal(shape=[4, 100, 512], dtype=tf.float32)
    encoder = TacotronEncoder(hidden_dim,
                              num_conv_layers,
                              conv_kernel_size,
                              conv_dilation_rate,
                              out_dim,
                              use_bias,
                              dropout)

    out = encoder(fake_input_1d)
    assert np.array_equal(tf.keras.backend.int_shape(out), [4, 100, out_dim])
