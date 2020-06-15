import tensorflow as tf
from tensorflow_tts.flows.utils.squeeze2D import Squeeze2D


class Squeeze2DTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.squeeze = Squeeze2D()

    def testSqueezeWithOutAnythng(self):
        x = tf.random.normal([32, 16, 8])
        y = self.squeeze(x, inverse=False)
        rev_x = self.squeeze(y, inverse=True)
        self.assertAllEqual(x, rev_x)
        zaux = tf.random.normal([32, 16, 16])
        y, new_zaux = self.squeeze(x, zaux=zaux, inverse=False)
        rev_x, rev_zaux = self.squeeze(y, zaux=new_zaux, inverse=True)
        self.assertAllEqual(x, rev_x)
        self.assertAllEqual(zaux, rev_zaux)
