from __future__ import absolute_import, division, print_function

import tensorflow as tf

from .base import BaseModel
from util.decorators import overrides
from util import const

Bernoulli = tf.contrib.distributions.Bernoulli
Normal    = tf.contrib.distributions.Normal
KL        = tf.contrib.distributions.kl

READ_N  = 12 # read glimpse grid width/height
WRITE_N = 12 # write glimpse grid width/height

ATTN = True

READ_ATTN  = ATTN
WRITE_ATTN = ATTN

ENC_SIZE = 256
DEC_SIZE = 256

T = 10

LSTM_ENC = tf.contrib.rnn.LSTMCell(ENC_SIZE, state_is_tuple=True)
LSTM_DEC = tf.contrib.rnn.LSTMCell(DEC_SIZE, state_is_tuple=True)


def encode(x, state, reuse):
  with tf.variable_scope('encode', reuse=reuse):
    return LSTM_ENC(x, state)


def decode(z, state, reuse):
  with tf.variable_scope('decode', reuse=reuse):
    return LSTM_ENC(z, state)


class DRAW(BaseModel):

  def __init__(self, *args, **kwargs):
    img_size = args[2] ** 2

    self.read_size = 2 * READ_N ** 2 if READ_ATTN else 2 * img_size
    self.write_size = WRITE_N ** 2 if WRITE_ATTN else img_size

    self.read = self.read_attn if READ_ATTN else self.read_no_attn
    self.write = self.write_attn if WRITE_ATTN else self.write_no_attn

    BaseModel.__init__(self, *args, **kwargs)

  def linear(self, x, output_dim):
    """
    Affine transformation `y = Wx + b`
    assumes x.shape = (batch_size, num_features)
    """
    w = tf.get_variable("w", [x.get_shape()[1], output_dim])
    b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, w) + b

  def read_no_attn(self, x, x_hat, h_dec_prev, reuse):
    return tf.concat([x, x_hat], 1)

  def filterbank(self, gx, gy, sigma2, delta, n):
    grid_i = tf.reshape(tf.cast(tf.range(n), tf.float32), [1, -1])
    mu_x = gx + (grid_i - n / 2 - 0.5) * delta  # eq 19
    mu_y = gy + (grid_i - n / 2 - 0.5) * delta  # eq 20
    a = tf.reshape(tf.cast(tf.range(self._input_size), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(self._input_size), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, n, 1])
    mu_y = tf.reshape(mu_y, [-1, n, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    fx = tf.exp(-tf.square((a - mu_x) / (2 * sigma2)))  # 2*sigma2?
    fy = tf.exp(-tf.square((b - mu_y) / (2 * sigma2)))  # batch x N x B
    # normalize, sum over A and B dims
    fx = fx / tf.maximum(tf.reduce_sum(fx, 2, keep_dims=True), const.EPS)
    fy = fy / tf.maximum(tf.reduce_sum(fy, 2, keep_dims=True), const.EPS)
    return fx, fy

  def attn_window(self, scope, h_dec, n, reuse):
    with tf.variable_scope(scope, reuse=reuse):
      params = self.linear(h_dec, 5)
    gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(params, 5, 1)
    gx = (self._input_size + 1) / 2 * (gx_ + 1)
    gy = (self._input_size + 1) / 2 * (gy_ + 1)
    sigma2 = tf.exp(log_sigma2)
    delta = (self._input_size - 1) / (n - 1) * tf.exp(log_delta)  # batch x N
    return self.filterbank(gx, gy, sigma2, delta, n) + (tf.exp(log_gamma),)

  def read_attn(self, x, x_hat, h_dec_prev, reuse):
    fx, fy, gamma = self.attn_window("read", h_dec_prev, READ_N, reuse)

    def filter_img(img, fx, fy, gamma, N):
      fxt = tf.transpose(fx, perm=[0, 2, 1])
      img = tf.reshape(img, [-1, self._input_size, self._input_size])
      glimpse = tf.matmul(fy, tf.matmul(img, fxt))
      glimpse = tf.reshape(glimpse, [-1, N * N])
      return glimpse * tf.reshape(gamma, [-1, 1])

    x = filter_img(x, fx, fy, gamma, READ_N)  # batch x (read_n*read_n)
    x_hat = filter_img(x_hat, fx, fy, gamma, READ_N)
    return tf.concat([x, x_hat], 1)  # concat along feature axis

  def sample_prior(self, e):
    return self._prior_mean + self._prior_var * e

  def sample_q(self, h_enc, e, reuse):
    """
    Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
    mu is (batch,z_size)
    """
    with tf.variable_scope("mu", reuse=reuse):
      mu = self.linear(h_enc, self._latent_dim)
    with tf.variable_scope("sigma", reuse=reuse):
      logsigma = self.linear(h_enc, self._latent_dim)
      sigma    = tf.exp(logsigma)
    return mu + sigma * e, mu, logsigma, sigma

  def write_no_attn(self, h_dec, reuse):
    with tf.variable_scope("write", reuse=reuse):
      return self.linear(h_dec, self._img_size)

  def write_attn(self, h_dec, reuse):
    num_samples = int(h_dec.get_shape()[0])

    with tf.variable_scope("writeW", reuse=reuse):
      w = self.linear(h_dec, self.write_size)

    w = tf.reshape(w, [num_samples, WRITE_N, WRITE_N])
    fx, fy, gamma = self.attn_window("write", h_dec, WRITE_N, reuse)
    fyt = tf.transpose(fy, perm=[0, 2, 1])
    wr = tf.matmul(fyt, tf.matmul(w, fx))
    wr = tf.reshape(wr, [num_samples, self._img_size])

    return wr * tf.reshape(1.0 / gamma, [-1, 1])

  def binary_crossentropy(self, t, o):
    return -(t * tf.log(o + const.EPS) + (1.0 - t) * tf.log(1.0 - o + const.EPS))

  @overrides(BaseModel)
  def _model(self, x, init_reuse=None):

    num_samples = int(x.shape[0])
    e = tf.random_normal((num_samples, self._latent_dim), mean=0, stddev=1)

    cs = [0] * T  # sequence of canvases

    # Gaussian params generated by SampleQ. We will need these for computing loss.
    mus, logsigmas, sigmas = [0] * T, [0] * T, [0] * T

    # Initial states
    h_dec_prev = tf.zeros((num_samples, DEC_SIZE))
    enc_state = LSTM_ENC.zero_state(num_samples, tf.float32)
    dec_state = LSTM_DEC.zero_state(num_samples, tf.float32)

    reuse = init_reuse

    # Construct the unrolled computational graph
    for t in range(T):
      c_prev = cs[t - 1] if t > 0 else tf.zeros((num_samples, self._img_size))
      x_hat = x - tf.sigmoid(c_prev)  # error image
      r = self.read(x, x_hat, h_dec_prev, reuse)

      # Encode
      h_enc, enc_state = encode(tf.concat([r, h_dec_prev], 1), enc_state, reuse)
      z, mus[t], logsigmas[t], sigmas[t] = self.sample_q(h_enc, e, reuse)

      # Decode
      h_dec, dec_state = decode(z, dec_state, reuse)
      cs[t] = c_prev + self.write(h_dec, reuse)  # store results
      h_dec_prev = h_dec

      reuse = True  # from now on, share variables

    # Reconstruction term appears to have been collapsed down to a single scalar
    # value (rather than one per item in minibatch).
    rec_x = tf.nn.sigmoid(cs[-1])

    # After computing binary cross entropy, sum across features then take the mean
    # of those sums across minibatches.
    rec_loss = tf.reduce_sum(self.binary_crossentropy(x, rec_x), 1)  # reconstruction term
    rec_loss = tf.reduce_mean(rec_loss)

    kl_terms = [0] * T
    for t in range(T):
      mu2 = tf.square(mus[t])
      sigma2 = tf.square(sigmas[t])
      kl_terms[t] = 0.5 * (tf.reduce_sum(mu2 + sigma2 - tf.log(sigma2), 1) - T)
    lat_loss = tf.reduce_mean(tf.add_n(kl_terms))  # average over minibatches

    self._loss = loss = rec_loss + lat_loss

    return rec_x, loss, rec_loss, lat_loss

  # @overrides(BaseModel)
  # def _build_train_op(self, x):
  #   rec_x, loss, rec_loss, lat_loss = self._model(x, init_reuse=None)
  #   self._losses.append(loss)
  #
  #   # optimizer = tf.train.AdamOptimizer(self._lr, beta1=0.5)
  #   # grads = optimizer.compute_gradients(loss)
  #   # for i, (g, v) in enumerate(grads):
  #   #   if g is not None:
  #   #     grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
  #   # self._train_op = optimizer.apply_gradients(grads)
  #
  #   return loss

  @overrides(BaseModel)
  def _build_prior_sample_op(self):
    e = tf.random_normal((self._num_samples, self._latent_dim), mean=0, stddev=1)

    cs = [0] * T  # sequence of canvases

    dec_state = LSTM_DEC.zero_state(self._num_samples, tf.float32)

    reuse = True

    # Construct the unrolled computational graph
    for t in range(T):
      c_prev = cs[t - 1] if t > 0 else tf.zeros((self._num_samples, self._img_size))

      z = self.sample_prior(e)

      # Decode
      h_dec, dec_state = decode(z, dec_state, reuse)
      cs[t] = c_prev + self.write(h_dec, reuse)  # store results

    self._samples = tf.nn.sigmoid(cs[-1])

  @overrides(BaseModel)
  def get_runname(self):
    template = "DRAW_%d_%.0E_%g_%g_%d_%d_%d_%d_%d_%d" % (
      self._latent_dim, self._lr_pure, self._prior_mean, self._prior_var,
      ATTN, T, READ_N, WRITE_N, ENC_SIZE, DEC_SIZE
    )
    return template

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
