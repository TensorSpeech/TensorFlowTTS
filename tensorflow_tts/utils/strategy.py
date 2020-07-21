# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
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
"""Strategy util functions"""
import tensorflow as tf


def return_strategy():
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) == 0:
        return tf.distribute.OneDeviceStrategy(device="/cpu:0")
    elif len(physical_devices) == 1:
        return tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        return tf.distribute.MirroredStrategy()


def calculate_3d_loss(y_gt, y_pred, loss_fn):
    """Calculate 3d loss, normally it's mel-spectrogram loss."""
    y_gt_T = tf.shape(y_gt)[1]
    y_pred_T = tf.shape(y_pred)[1]

    # there is a mismath length when training multiple GPU.
    # we need slice the longer tensor to make sure the loss
    # calculated correctly.
    if y_gt_T > y_pred_T:
        y_gt = tf.slice(y_gt, [0, 0, 0], [-1, y_pred_T, -1])
    elif y_pred_T > y_gt_T:
        y_pred = tf.slice(y_pred, [0, 0, 0], [-1, y_gt_T, -1])

    loss = loss_fn(y_gt, y_pred)
    if isinstance(loss, tuple) is False:
        loss = tf.reduce_mean(loss, list(range(1, len(loss.shape))))  # shape = [B]
    else:
        loss = list(loss)
        for i in range(len(loss)):
            loss[i] = tf.reduce_mean(
                loss[i], list(range(1, len(loss[i].shape)))
            )  # shape = [B]
    return loss


def calculate_2d_loss(y_gt, y_pred, loss_fn):
    """Calculate 2d loss, normally it's durrations/f0s/energys loss."""
    y_gt_T = tf.shape(y_gt)[1]
    y_pred_T = tf.shape(y_pred)[1]

    # there is a mismath length when training multiple GPU.
    # we need slice the longer tensor to make sure the loss
    # calculated correctly.
    if y_gt_T > y_pred_T:
        y_gt = tf.slice(y_gt, [0, 0], [-1, y_pred_T])
    elif y_pred_T > y_gt_T:
        y_pred = tf.slice(y_pred, [0, 0], [-1, y_gt_T])

    loss = loss_fn(y_gt, y_pred)
    if isinstance(loss, tuple) is False:
        loss = tf.reduce_mean(loss, list(range(1, len(loss.shape))))  # shape = [B]
    else:
        loss = list(loss)
        for i in range(len(loss)):
            loss[i] = tf.reduce_mean(
                loss[i], list(range(1, len(loss[i].shape)))
            )  # shape = [B]

    return loss
