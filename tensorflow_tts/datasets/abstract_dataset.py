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
"""Abstract Dataset modules."""

import abc

import tensorflow as tf


class AbstractDataset(metaclass=abc.ABCMeta):
    """Abstract Dataset module for Dataset Loader."""

    @abc.abstractmethod
    def get_args(self):
        """Return args for generator function."""
        pass

    @abc.abstractmethod
    def generator(self):
        """Generator function, should have args from get_args function."""
        pass

    @abc.abstractmethod
    def get_output_dtypes(self):
        """Return output dtypes for each element from generator."""
        pass

    @abc.abstractmethod
    def get_len_dataset(self):
        """Return number of samples on dataset."""
        pass

    def create(
        self,
        allow_cache=False,
        batch_size=1,
        is_shuffle=False,
        map_fn=None,
        reshuffle_each_iteration=True,
    ):
        """Create tf.dataset function."""
        output_types = self.get_output_dtypes()
        datasets = tf.data.Dataset.from_generator(
            self.generator, output_types=output_types, args=(self.get_args())
        )

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        if batch_size > 1 and map_fn is None:
            raise ValueError("map function must define when batch_size > 1.")

        if map_fn is not None:
            datasets = datasets.map(map_fn, tf.data.experimental.AUTOTUNE)

        datasets = datasets.batch(batch_size)
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)

        return datasets
