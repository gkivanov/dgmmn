from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc, stack
from .vae import VAE
from util.spatial_transformer import transformer
from util.decorators import overrides
from util import const
import numpy as np

FARCH = [250, 500, 1000]
RARCH = list(reversed(FARCH))
TRSHAPE = (100, 100)  # transformed shape
THDIM = 6  # theta dim


class STNFullyConnectedVAE(VAE):

  @overrides(VAE)
  def encode(self, x, reuse=None):
    """
    Encoder network of multiple fully connected layers, followed by a 
    spatial transformer network (STN), which is then followed by another set of
    fully connected layers, which perform the actual encoding task.

    :param x: input tensor
    :param reuse: scope reuse flag
    :return: tuple representing the mean and variance of the variational posterior
    """
    with tf.variable_scope(const.SCOPE_ENC, reuse=reuse):
      # First set of FC layers
      theta = stack(x, fc, FARCH, activation_fn=tf.nn.tanh,
                    scope='stn' + const.SCOPE_ENC + '1', reuse=reuse)

      theta = fc(theta, THDIM, activation_fn=None,
                 scope='stn' + const.SCOPE_ENC + '1', reuse=reuse)

      # Transform
      x_expanded = tf.reshape(x, [-1, self._input_size, self._input_size, 1])
      x_transformed = transformer(x_expanded, theta, TRSHAPE)

      x_transformed_flat = tf.reshape(x_transformed, [-1, np.prod(TRSHAPE)])

      # Encode
      encoded = stack(x_transformed_flat, fc, FARCH, activation_fn=tf.nn.tanh,
                      scope=const.SCOPE_ENC + '1', reuse=reuse)

      encoded = fc(encoded, self._latent_dim * 2, activation_fn=None,
                   scope=const.SCOPE_ENC + '2', reuse=reuse)

      mu  = encoded[:, :self._latent_dim]
      var = tf.nn.softplus(encoded[:, self._latent_dim:])

      return mu, var

  # ----------------------------------------------------------------------------

  @overrides(VAE)
  def decode(self, z, reuse=None):
    """
    Decoder network of multiple fully connected layer, followed by an STN.

    :param z: sample from the variational posterior Q
    :param reuse: scope reuse flag
    :return: the logits expressing the network itself
    """
    with tf.variable_scope(const.SCOPE_DEC, reuse=reuse):
      theta = z[:, :THDIM]
      z_enc = z[:, THDIM:]

      # Decode
      net_decode = stack(z_enc, fc, RARCH, activation_fn=tf.nn.tanh,
                         scope=const.SCOPE_DEC + '1', reuse=reuse)

      net_decode = fc(net_decode, np.prod(TRSHAPE), activation_fn=None,
                      scope=const.SCOPE_DEC + '2', reuse=reuse)

      net_decode_square = tf.reshape(net_decode, [-1] + list(TRSHAPE) + [1])
      d = self._input_size

      # Transform
      transformed = transformer(net_decode_square, theta, [d, d])
      return tf.reshape(transformed, [-1, d * d])

  # ----------------------------------------------------------------------------

  @overrides(VAE)
  def get_runname(self):
    template = "STNFCVAE_%d_%.0E_%g_%g_%s" % (
      self._latent_dim, self._lr_pure, self._prior_mean, self._prior_var,
      ",".join(map(str, FARCH))
    )
    return template
