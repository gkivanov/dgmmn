"""Functions for loading and reading DGMMN data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import math
import os
from itertools import tee
from random import shuffle

import numpy as np
from scipy.misc import imread
from tensorflow.python.framework import dtypes


def generator(paths, batch_size):
  batch_imgs, batch_lbls = [], []
  size = 0

  for path in paths:
    batch_imgs.append(imread(path))

    batch_lbls.append(1 if "clean" in path else 0)

    size += 1

    if size >= batch_size:
      yield np.expand_dims(np.array(batch_imgs), axis=3), batch_lbls
      size = 0
      batch_imgs, batch_lbls = [], []


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class DGMMN:

  def __init__(self, data_dir, batch_size, val_size=0.2,
               dtype=dtypes.float32, reshape=True):
    self._reshape = reshape
    self._dtype = dtype

    self.__train_epochs_completed = 0
    self.__val_epochs_completed   = 0

    cleanlist = glob.glob(os.path.join(data_dir, "clean", "*"))
    dirtylist = glob.glob(os.path.join(data_dir, "dirty", "*"))

    filelist = cleanlist + dirtylist

    self._total_images = len(filelist)

    shuffle(filelist)

    validx = int(len(filelist) * (1 - val_size))

    trainbars = filelist[:validx]
    valbars = filelist[validx:]

    self._tr_batches_per_epoch = int(math.floor(len(trainbars) / batch_size))
    self._val_batches_per_epoch = int(math.floor(len(valbars) / batch_size))

    if len(trainbars) < batch_size:
      raise ValueError("Training Set too small (%d) for batch_size (%d)." %
                       (len(trainbars), batch_size))

    if len(valbars) < batch_size:
      raise ValueError("Validation Set too small (%d) for batch_size (%d)." %
                       (len(valbars), batch_size))

    self.train, self.nxttrain = tee(generator(trainbars, batch_size))
    self.validate, self.nxtval = tee(generator(valbars, batch_size))

    imgs, lbls = [], []
    for path in valbars:
      imgs.append(imread(path))
      lbls.append(1 if "clean" in path else 0)
    imgs = np.expand_dims(np.array(imgs), axis=3)
    self.validation = self.parse((imgs, lbls))

  @property
  def total_images(self):
    return self._total_images

  @property
  def train_batches_per_epoch(self):
    return self._tr_batches_per_epoch

  @property
  def val_batches_per_epoch(self):
    return self._val_batches_per_epoch

  def _reformat(self, imgs):
    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if self._reshape:
      assert imgs.shape[3] == 1
      imgs = imgs.reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2])

    if self._dtype == dtypes.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      imgs = imgs.astype(np.float32)
      imgs = np.multiply(imgs, 1.0 / 255.0)

    return imgs

  def parse(self, batch):
    imgs, lbls = batch
    imgs = self._reformat(imgs)
    lbls = dense_to_one_hot(np.array(lbls), 2)
    return imgs, lbls

  def next_train_batch(self):
    """Return the next `batch_size` training examples from this data set."""
    try:
      return self.parse(next(self.train))
    except StopIteration:
      self.__train_epochs_completed += 1
      self.train, self.nxttrain = tee(self.nxttrain)
      return self.parse(next(self.train))

  def next_validate_batch(self):
    """Return the next `batch_size` validation examples from this data set."""
    try:
      return self.parse(next(self.validate))
    except StopIteration:
      self.__val_epochs_completed += 1
      self.validate, self.nxtval = tee(self.nxtval)
      return self.parse(next(self.validate))
