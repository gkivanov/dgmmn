from __future__ import absolute_import, division, print_function

from .vae import VAE
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
from util.decorators import overrides
from util import const


class LinearVAE(VAE):

  @overrides(VAE)
  def encode(self, x, reuse):
    """
    Encoder network of a single fully connected layer.
    
    :param x: input tensor
    :param reuse: scope reuse flag
    :return: tuple representing the mean and variance of the variational posterior
    """
    with tf.variable_scope(const.SCOPE_ENC, reuse=reuse):
      z = fc(x, self._latent_dim * 2, activation_fn=None)

      mu  = z[:, :self._latent_dim]
      var = tf.nn.softplus(z[:, self._latent_dim:])

      return mu, var

  # ----------------------------------------------------------------------------

  @overrides(VAE)
  def decode(self, z, reuse):
    """
    Decoder network of a single fully connected layer.
    
    :param z: sample from the variational posterior Q
    :param reuse: scope reuse flag
    :return: the logits expressing the network itself
    """
    with tf.variable_scope(const.SCOPE_DEC, reuse=reuse):
      return fc(z, self._img_size, activation_fn=None)

  # ----------------------------------------------------------------------------

  @overrides(VAE)
  def get_runname(self):
    template = "LinVAE_%d_%.0E_%g_%g" % (
      self._latent_dim, self._lr_pure, self._prior_mean, self._prior_var,
    )
    return template
