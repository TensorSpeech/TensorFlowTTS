#!/usr/bin/env python3
import tenosrflow as tf
from TFGENZOO.flows.utils import gausiann_likelihood


def nll(z: tf.Tensor, mask: tf.Tensor):
    """negative log likelihood for z

    Args:
       z (tf.Tensor): base latent variable [B, T, C]
    Returns:
       tf.Tensor: nll [B, T, C]
    """
    ll = gausiann_likelihood(tf.zeros(tf.shape(z)), tf.zeros(tf.shape(z)), z)
    return ll * tf.cast(mask, z.dtype)
