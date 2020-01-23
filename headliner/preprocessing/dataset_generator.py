from typing import Iterable, Callable

import tensorflow as tf


class DatasetGenerator:

    def __init__(self,
                 batch_size: int,
                 shuffle_buffer_size=None,
                 rank=2):
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        if rank == 2:
            self.tensor_types = (tf.int32, tf.int32)
            self.tensor_shapes = (tf.TensorShape([None]), tf.TensorShape([None]))
        elif rank == 3:
            self.tensor_types = (tf.int32, tf.int32, tf.int32)
            self.tensor_shapes = (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]))
        else:
            raise ValueError('Rank must be either 2 or 3, but was: {}'.format(rank))

    def __call__(self, data_generator_func: Callable[..., Iterable]) -> tf.data.Dataset:
        """
        Initializes a dataset generator.

        Args:
            data_generator_func: Callable that returns an iterable over the data to be batched, e.g. lambda: [1, 2, 3].
        """

        dataset = tf.data.Dataset.from_generator(data_generator_func,
                                                 self.tensor_types,
                                                 self.tensor_shapes)
        if self.shuffle_buffer_size is not None:
            dataset = dataset.shuffle(self.shuffle_buffer_size)
        dataset = dataset.padded_batch(batch_size=self.batch_size,
                                       padded_shapes=self.tensor_shapes,
                                       drop_remainder=True)
        return dataset
