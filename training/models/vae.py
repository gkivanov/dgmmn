from __future__ import absolute_import, division, print_function

import tensorflow as tf

from .base import BaseModel
from util.decorators import overrides
from abc import ABCMeta, abstractmethod
from util.divergences import kl
from util.const import EPS

from tensorflow.contrib.distributions import Bernoulli, Normal, kl as KL


class VAE(BaseModel):
  __metaclass__ = ABCMeta

  def __init__(self, *args, **kwargs):
    BaseModel.__init__(self, *args, **kwargs)

  # ----------------------------------------------------------------------------

  @abstractmethod
  def encode(self, x, reuse): pass

  @abstractmethod
  def decode(self, z, reuse): pass

  # ----------------------------------------------------------------------------

  @overrides(BaseModel)
  def _model(self, input_tensor, reuse):
    with tf.name_scope("model"):
      # Encode
      z_mu, z_var = self.encode(input_tensor, reuse)
      z_var = tf.maximum(tf.constant(0.0, dtype=tf.float32), z_var) + EPS
      z_stddev = tf.sqrt(z_var)

      q_z = Normal(z_mu, z_stddev)  # variational posterior

      z = q_z.sample()  # b, latent_dim

      # Reconstruct/decode
      logits_x = self.decode(z, reuse)

      rec_x = tf.nn.sigmoid(logits_x)
      p_x = Bernoulli(logits=logits_x)

      # Loss
      rec_loss = -tf.reduce_sum(p_x.log_prob(input_tensor), 1)
      lat_loss = kl(z_mu, z_var, self._prior_mean, self._prior_var)
      lat_loss = tf.reduce_sum(lat_loss, 1)
      loss = tf.reduce_mean(lat_loss + rec_loss)

      return rec_x, loss, rec_loss, lat_loss

  # ----------------------------------------------------------------------------

  @overrides(BaseModel)
  def _build_prior_sample_op(self):
    """ Builds the Sampling operation """
    num_samples = self._num_samples

    # Generate some samples from prior
    z_samples = tf.random_normal([num_samples, self._latent_dim])
    logits_samples = self.decode(z_samples, True)

    self._samples = tf.nn.sigmoid(logits_samples)

  # ----------------------------------------------------------------------------

  @overrides(BaseModel)
  def _build_train_op(self, x):
    """ Builds the Training operation """
    rec_x, loss, rec_loss, lat_loss = self._model(x, None)
    self._losses.append(loss)
    return [loss]

  # ----------------------------------------------------------------------------

  @overrides(BaseModel)
  def init_optimizers(self):
    self._optimizers.append(tf.train.AdamOptimizer(self._lr))
