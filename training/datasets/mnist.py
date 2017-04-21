"""Functions for loading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import dtypes

from .base import BaseDataset
from util.decorators import overrides


class MNIST(BaseDataset):

  @property
  def train_batches_per_epoch(self):
    return 1000

  @property
  def val_batches_per_epoch(self):
    return 100

  @property
  def epochs_completed(self):
    return self.train.epochs_completed

  def __init__(self, data_dir, batch_size, dtype=dtypes.float32, reshape=True):
    BaseDataset.__init__(self, batch_size, dtype, reshape)
    ds = input_data.read_data_sets(data_dir, one_hot=True)

    self.train    = ds.train
    self.validate = ds.validation

  @overrides(BaseDataset)
  def next_train_batch(self):
    return self.train.next_batch(self._batch_size)[0]

  @overrides(BaseDataset)
  def next_validate_batch(self):
    return self.validate.next_batch(self._batch_size)[0]

  @overrides(BaseDataset)
  def get_runname(self):
    return "MNIST"
