import logging
import os
import pytest
import tensorflow as tf

from tensorflow_tts.models import TFFastSpeech

os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")


class FastSpeechConfig(object):
    """Initialize FastSpeech Config."""

    def __init__(
            self,
            vocab_size=200,
            n_speakers=1,
            hidden_size=384,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            intermediate_kernel_size=3,
            num_duration_conv_layers=2,
            duration_predictor_filters=128,
            duration_predictor_kernel_sizes=3,
            num_mels=80,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            duration_predictor_dropout_probs=0.1,
            max_position_embeddings=2048,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            output_attentions=False,
            output_hidden_states=False):
        """Init parameters for Fastspeech model."""
        self.vocab_size = vocab_size
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
        self.num_duration_conv_layers = num_duration_conv_layers
        self.duration_predictor_filters = duration_predictor_filters
        self.duration_predictor_kernel_sizes = duration_predictor_kernel_sizes
        self.num_mels = num_mels


@pytest.mark.parametrize(
    "num_hidden_layers,n_speakers", [
        (2, 1), (3, 2), (4, 3)
    ]
)
def test_fastspeech_trainable(num_hidden_layers, n_speakers):
    config = FastSpeechConfig(num_hidden_layers=num_hidden_layers, n_speakers=n_speakers)

    fastspeech = TFFastSpeech(config, name='fastspeech')
    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    # fake inputs
    input_ids = tf.convert_to_tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], tf.int32)
    attention_mask = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)
    speaker_ids = tf.convert_to_tensor([0], tf.int32)
    duration_gts = tf.convert_to_tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], tf.int32)

    mel_gts = tf.random.uniform(shape=[1, 10, 80], dtype=tf.float32)

    @tf.function
    def one_step_training():
        with tf.GradientTape() as tape:
            mel_outputs, duration_outputs = fastspeech(
                input_ids, attention_mask, speaker_ids, duration_gts, training=True)
            duration_loss = tf.keras.losses.MeanSquaredError()(duration_gts, duration_outputs)
            mel_loss = tf.keras.losses.MeanSquaredError()(mel_gts, mel_outputs)
            loss = duration_loss + mel_loss
        gradients = tape.gradient(loss, fastspeech.trainable_variables)
        optimizer.apply_gradients(zip(gradients, fastspeech.trainable_variables))

        tf.print(loss)

    import time
    for i in range(100):
        if i == 1:
            start = time.time()
        one_step_training()
    print(time.time() - start)
