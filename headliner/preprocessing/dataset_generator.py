from typing import Iterable, Callable

import tensorflow as tf


class DatasetGenerator:

    def __init__(self,
                 batch_size: int,
                 shuffle_buffer_size=None):
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size

    def __call__(self, data_generator_func: Callable[..., Iterable]) -> tf.data.Dataset:
        """
        Initializes a dataset generator.

        Args:
            data_generator_func: Callable that returns an iterable over the data to be batched, e.g. lambda: [1, 2, 3].
        """
        tensor_types = (tf.int32, tf.int32)
        tensor_shapes = (tf.TensorShape([None]), tf.TensorShape([None]))
        dataset = tf.data.Dataset.from_generator(data_generator_func, tensor_types, tensor_shapes)
        if self.shuffle_buffer_size is not None:
            dataset = dataset.shuffle(self.shuffle_buffer_size)
        dataset = dataset.padded_batch(batch_size=self.batch_size,
                                       padded_shapes=([None], [None]),
                                       drop_remainder=True)
        return dataset
