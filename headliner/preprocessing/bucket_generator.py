from random import Random
from typing import Iterable, Callable, Iterator, List


class BucketGenerator:
    """
    Performs bucketing of elements in a dataset by length.
    """

    def __init__(self,
                 element_length_function: Callable[..., int],
                 batch_size: int,
                 buffer_size_batches=1000,
                 batches_to_bucket=10,
                 shuffle=True,
                 seed=None) -> None:
        """
        Initializes the BucketGenerator.

        Args:
            element_length_function: Element_length_function: function from element in the dataset to int that
            determines the length of the element.
            batch_size: The size of the batches to bucket the sequences into
                buffer_size_batches: buffer_size_batches: number of batches to keep in internal memory.
            batches_to_bucket: Number of batches in buffer to use for bucketing.
                If set to buffer_size_batches, the resulting batches will be deterministic.
            shuffle: Whether to shuffle elements across batches and the resulting buckets.
            seed: Seed for shuffling.
        """

        self.sequence_length_function = element_length_function
        self.batch_size = batch_size
        self.buffer_size_batches = buffer_size_batches
        self.batches_to_shuffle = batches_to_bucket
        self.shuffle = shuffle
        self.random = Random(seed)

    def __call__(self, data: Iterable) -> Iterable:
        """
        Returns iterable of data with elements ordered by bucketed sequence lengths, e.g for batch size = 2 the
        transformation could look like this:
        [1], [3, 3, 3], [1], [4, 4, 4, 4] -> [1], [1], [3, 3, 3], [4, 4, 4, 4]
        """
        data_iter = iter(data)
        bucket_gen = self._generate_buckets(data_iter)
        return bucket_gen

    def _generate_buckets(self, data_iter: Iterator) -> List[List]:
        buffered_data = self._fetch_buffered_data(data_iter)
        while len(buffered_data) > 0:
            buckets = self._to_buckets(buffered_data)
            for bucket in buckets:
                for element in bucket:
                    yield element
            buffered_data = self._fetch_buffered_data(data_iter)
            del buckets

    def _to_buckets(self, buffered_data: List) -> List[List]:
        self._shuffle_if_required(buffered_data)
        buffered_data = self._sort_blocks(buffered_data)
        buckets = []
        for i in range(0, len(buffered_data), self.batch_size):
            bucket = buffered_data[i:i + self.batch_size]
            if len(bucket) == self.batch_size:
                buckets.append(bucket)
        self._shuffle_if_required(buckets)
        return buckets

    def _sort_blocks(self, buffered_data: List) -> List:
        block_size = self.batches_to_shuffle * self.batch_size
        buffered_data_sorted = []
        for i in range(0, len(buffered_data), block_size):
            sorted_block = buffered_data[i:i + block_size]
            sorted_block.sort(key=self.sequence_length_function)
            buffered_data_sorted.extend(sorted_block)
        return buffered_data_sorted

    def _fetch_buffered_data(self, data_iter: Iterator):
        buffered_data = []
        for _ in range(self.buffer_size_batches * self.batch_size):
            try:
                buffered_data.append(next(data_iter))
            except StopIteration:
                pass
        return buffered_data

    def _shuffle_if_required(self, list_to_shuffle):
        if self.shuffle:
            self.random.shuffle(list_to_shuffle)
