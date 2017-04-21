from __future__ import absolute_import, division, print_function

import tensorflow as tf

from .base import BaseModel
from util.decorators import overrides
from abc import ABCMeta, abstractmethod
from tensorflow.contrib.losses import sigmoid_cross_entropy


class GAN(BaseModel):
  __metaclass__ = ABCMeta

  def __init__(self, *args, **kwargs):
    BaseModel.__init__(self, *args, **kwargs)

  # ----------------------------------------------------------------------------

  @abstractmethod
  def discriminator(self, x, reuse): pass

  @abstractmethod
  def generator(self, z, reuse): pass

  # ----------------------------------------------------------------------------

  @overrides(BaseModel)
  def init_optimizers(self):
    self._optimizers.append(tf.train.AdamOptimizer(self._lr))
    self._optimizers.append(tf.train.AdamOptimizer(self._lr))

  def _build_train_op(self, x):
    d_loss, g_loss = self._model(x, None)

    self._losses.append(d_loss)

    return [d_loss, g_loss]

  @overrides(BaseModel)
  def _build_prior_sample_op(self):
    num_samples = self._num_samples

    # Generate some samples from prior
    z_samples = tf.random_normal([num_samples, self._latent_dim])
    logits_samples = self.generator(z_samples, True)

    self._samples = tf.nn.sigmoid(logits_samples)

  @overrides(BaseModel)
  def _model(self, x, reuse):
    with tf.name_scope("model"):
      disc1 = self.discriminator(x, reuse)  # positive examples

      z = tf.random_normal([int(x.shape[0]), self._latent_dim])

      logits_x = self.generator(z, reuse)

      disc2 = self.discriminator(logits_x, True)  # generated examples

      d_loss = sigmoid_cross_entropy(disc1, tf.ones(tf.shape(disc1))) + \
               sigmoid_cross_entropy(disc2, tf.zeros(tf.shape(disc1)))

      g_loss = sigmoid_cross_entropy(disc2, tf.ones(tf.shape(disc2)))

      return d_loss, g_loss
