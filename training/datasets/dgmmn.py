"""Functions for loading and reading DGMMN data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import glob

import numpy as np

from PIL import Image
from tensorflow.python.framework import dtypes

from .base import BaseDataset
from util.decorators import overrides
from threading import Lock


def load(paths, dim):
  batch_size = len(paths)
  batch_imgs = np.empty((batch_size, dim * dim))
  for idx, p in enumerate(paths):
    batch_imgs[idx] = np.asarray(Image.open(p), np.uint8).flatten()
  return batch_imgs


class DGMMN(BaseDataset):

  def __init__(self, data_dir, batch_size, dtype=dtypes.float32, reshape=True):
    BaseDataset.__init__(self, batch_size, dtype, reshape)

    self.__train_epochs_completed = 0
    self.__val_epochs_completed   = 0

    # Indices for train and val set progress tracking
    self._train_index = 0
    self._val_index = 0

    # Lock objects for thread synchronization
    self._train_lock = Lock()
    self._val_lock = Lock()

    traindir = os.path.join(data_dir, "train", "*")
    valdir = os.path.join(data_dir, "test", "*")

    self.trainpths = trainpths = glob.glob(traindir)
    self.valpths   = valpths = glob.glob(valdir)

    self._total_images = len(trainpths) + len(valpths)
    self._tr_batches_per_epoch = int(math.floor(len(trainpths) / batch_size))
    self._val_batches_per_epoch = int(math.floor(len(valpths) / batch_size))

    if len(trainpths) < batch_size:
      raise ValueError("Training Set too small (%d) for batch_size (%d)." %
                       (len(trainpths), batch_size))

    if len(valpths) < batch_size:
      raise ValueError("Validation Set too small (%d) for batch_size (%d)." %
                       (len(valpths), batch_size))

  @property
  def train_batches_per_epoch(self):
    return self._tr_batches_per_epoch

  @property
  def val_batches_per_epoch(self):
    return self._val_batches_per_epoch

  def _reformat(self, imgs):
    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    # if self._reshape:
    #   assert imgs.shape[3] == 1
    #   imgs = imgs.reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2])

    if self._dtype == dtypes.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      imgs = imgs.astype(np.float32)
      imgs = np.multiply(imgs, 1.0 / 255.0)

    return imgs

  @overrides(BaseDataset)
  def next_train_batch(self):
    """Return the next `batch_size` training examples from this data set."""
    paths = []

    self._train_lock.acquire()

    for _ in range(self._batch_size):
      paths.append(self.trainpths[self._train_index % len(self.trainpths)])
      self._train_index += 1

    self._train_lock.release()

    return self._reformat(load(paths, 200))

  @overrides(BaseDataset)
  def next_validate_batch(self):
    """Return the next `batch_size` validation examples from this data set."""
    paths = []

    self._val_lock.acquire()

    for _ in range(self._batch_size):
      paths.append(self.valpths[self._val_index % len(self.valpths)])
      self._val_index += 1

    self._val_lock.release()

    return self._reformat(load(paths, 200))

  @overrides(BaseDataset)
  def get_runname(self):
    return "DGMMN_%d_%d" % (self._total_images, self._batch_size)
