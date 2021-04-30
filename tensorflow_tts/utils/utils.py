# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)
"""Utility functions."""

import fnmatch
import os
import re
import tempfile
from pathlib import Path

import tensorflow as tf

MODEL_FILE_NAME = "model.h5"
CONFIG_FILE_NAME = "config.yml"
PROCESSOR_FILE_NAME = "processor.json"
LIBRARY_NAME = "tensorflow_tts"
CACHE_DIRECTORY = os.path.join(Path.home(), ".cache", LIBRARY_NAME)


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.
    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.
    Returns:
        list: List of found filenames.
    """
    files = []
    for root, _, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def _path_requires_gfile(filepath):
    """Checks if the given path requires use of GFile API.

    Args:
        filepath (str): Path to check.
    Returns:
        bool: True if the given path needs GFile API to access, such as
            "s3://some/path" and "gs://some/path".
    """
    # If the filepath contains a protocol (e.g. "gs://"), it should be handled
    # using TensorFlow GFile API.
    return bool(re.match(r"^[a-z]+://", filepath))


def save_weights(model, filepath):
    """Save model weights.

    Same as model.save_weights(filepath), but supports saving to S3 or GCS
    buckets using TensorFlow GFile API.

    Args:
        model (tf.keras.Model): Model to save.
        filepath (str): Path to save the model weights to.
    """
    if not _path_requires_gfile(filepath):
        model.save_weights(filepath)
        return

    # Save to a local temp file and copy to the desired path using GFile API.
    _, ext = os.path.splitext(filepath)
    with tempfile.NamedTemporaryFile(suffix=ext) as temp_file:
        model.save_weights(temp_file.name)
        # To preserve the original semantics, we need to overwrite the target
        # file.
        tf.io.gfile.copy(temp_file.name, filepath, overwrite=True)


def load_weights(model, filepath):
    """Load model weights.

    Same as model.load_weights(filepath), but supports loading from S3 or GCS
    buckets using TensorFlow GFile API.

    Args:
        model (tf.keras.Model): Model to load weights to.
        filepath (str): Path to the weights file.
    """
    if not _path_requires_gfile(filepath):
        model.load_weights(filepath)
        return

    # Make a local copy and load it.
    _, ext = os.path.splitext(filepath)
    with tempfile.NamedTemporaryFile(suffix=ext) as temp_file:
        # The target temp_file should be created above, so we need to overwrite.
        tf.io.gfile.copy(filepath, temp_file.name, overwrite=True)
        model.load_weights(temp_file.name)
