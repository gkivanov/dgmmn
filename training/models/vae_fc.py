from __future__ import absolute_import, division, print_function

from .vae import VAE
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc, stack
from util.decorators import overrides
from util import const

FARCH = [250, 500, 1000]
RARCH = list(reversed(FARCH))


class FullyConnectedVAE(VAE):
  @overrides(VAE)
  def encode(self, x, reuse):
    """
    Encoder network of multiple fully connected layers.

    :param x: input tensor
    :param reuse: scope reuse flag
    :return: tuple with the mean and variance of the variational posterior
    """
    with tf.variable_scope(const.SCOPE_ENC, reuse=reuse):
      net = stack(x, fc, FARCH, scope=const.SCOPE_ENC + '1', reuse=reuse)
      z = fc(net, self._latent_dim * 2, activation_fn=None,
             scope=const.SCOPE_ENC + '2', reuse=reuse)

      mu  = z[:, :self._latent_dim]
      var = tf.nn.softplus(z[:, self._latent_dim:])

      return mu, var

  # ----------------------------------------------------------------------------

  @overrides(VAE)
  def decode(self, z, reuse):
    """
    Decoder network of multiple fully connected layer.

    :param z: sample from the variational posterior Q
    :param reuse: scope reuse flag
    :return: the logits expressing the network itself
    """
    with tf.variable_scope(const.SCOPE_DEC, reuse=reuse):
      net = stack(z, fc, RARCH, scope=const.SCOPE_DEC + '1', reuse=reuse)
      return fc(net, self._img_size, activation_fn=None,
                scope=const.SCOPE_DEC + '2', reuse=reuse)

  # ----------------------------------------------------------------------------

  @overrides(VAE)
  def get_runname(self):
    template = "FCVAE_%d_%.0E_%g_%g_%s" % (
      self._latent_dim, self._lr_pure, self._prior_mean, self._prior_var,
      ",".join(map(str, FARCH))
    )
    return template
