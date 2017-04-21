from __future__ import absolute_import, division, print_function

from .vae import VAE
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim \
  import fully_connected as fc, conv2d, conv2d_transpose as conv2dtr, max_pool2d
from util.decorators import overrides
from util import const


class ConvolutionalVAE(VAE):

  @overrides(VAE)
  def encode(self, x, reuse):
    """
    Encoder network of multiple convolutional layers and a fully connected
    one at the end.

    :param x: input tensor
    :param reuse: scope reuse flag
    :return: tuple with the mean and variance of the variational posterior
    """
    with tf.variable_scope(const.SCOPE_ENC, reuse=reuse):
      x = tf.reshape(x, [-1, self._input_size, self._input_size, 1])

      with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.elu,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params={'scale': True}):
        # b, 200, 200, 1
        net = conv2d(x, 32, 3, stride=2, reuse=reuse,
                     scope=const.SCOPE_ENC + "1")
        # b,100,100,32
        net = conv2d(net, 64, 3, stride=2, reuse=reuse,
                     scope=const.SCOPE_ENC + "2")
        net = max_pool2d(net, 2)
        # b,50,50,64
        net = conv2d(net, 128, 3, padding='VALID', reuse=reuse,
                     scope=const.SCOPE_ENC + "3")
        # b,21,21,128
        net = conv2d(net, 256, 3, padding='VALID', reuse=reuse,
                     scope=const.SCOPE_ENC + "4")
        # b,17,17,256
        net = max_pool2d(net, 2)
        net = conv2d(net, 256, 3, padding='VALID', reuse=reuse,
                     scope=const.SCOPE_ENC + "5")
        # b,15,15,256
        # net = slim.dropout(net, 0.9)
        net = slim.flatten(net)  # b,?
        z = fc(net, self._latent_dim * 2, activation_fn=None,
               reuse=reuse, scope=const.SCOPE_ENC + "6")  # b,2*latent_dum

      mu  = z[:, :self._latent_dim]
      var = tf.nn.softplus(z[:, self._latent_dim:])

      return mu, var

  # ----------------------------------------------------------------------------

  @overrides(VAE)
  def decode(self, z, reuse):
    """
    Decoder network of multiple deconvolutional layers.

    :param z: sample from the variational posterior Q
    :param reuse: scope reuse flag
    :return: the logits expressing the network itself
    """
    with tf.variable_scope(const.SCOPE_DEC, reuse=reuse):
      net = tf.reshape(z, [-1, 1, 1, self._latent_dim])
      with slim.arg_scope([slim.conv2d_transpose],
                          activation_fn=tf.nn.elu,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params={'scale': True}):
        # b,1,1,latent_dim
        net = conv2dtr(net, 256, 13, padding='VALID', reuse=reuse,
                       scope=const.SCOPE_DEC + "0")
        # b,13,13,256
        net = conv2dtr(net, 128, 13, padding='VALID', reuse=True,
                       scope=const.SCOPE_DEC + "1")
        # b,25,25,128
        net = conv2dtr(net, 64, 13, stride=2, reuse=True,
                       scope=const.SCOPE_DEC + "2")
        # b,50,50,64
        net = conv2dtr(net, 32, 13, stride=2, reuse=True,
                       scope=const.SCOPE_DEC + "3")
        # b,100,100,32
        net = conv2dtr(net, 1, 20, stride=2, activation_fn=None, reuse=True,
                       scope=const.SCOPE_DEC + "4")
        # b,200,200,1
        net = slim.flatten(net)
        # b, 40000
        return net

  # ----------------------------------------------------------------------------

  @overrides(VAE)
  def get_runname(self):
    template = "CVAE_%d_%.0E_%g_%g_%s" % (
      self._latent_dim, self._lr_pure, self._prior_mean, self._prior_var,
      ",".join(map(str, [3, 3, 'M', 3, 3, 'M', 3, 'R', 13, 13, 13, 13, 20]))
    )
    return template
