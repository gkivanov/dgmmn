from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

import tensorflow as tf

from util import const, gradients


class BaseModel(object):
  __metaclass__ = ABCMeta

  def __init__(self, train, validate, input_size, latent_dim, learning_rate,
               samples, num_gpus, prior_mean, prior_var):

    self._losses = []
    self._sample_op = None
    self._eval_op = None
    self._reconstr_op = None
    self._samples = None

    self._num_gpus = num_gpus
    self._input_size = input_size
    self._img_size = input_size ** 2
    self._num_samples = samples
    self._lr_pure = learning_rate
    self._lr = tf.constant(learning_rate)
    self._latent_dim = latent_dim

    self._prior_mean = float(prior_mean)
    self._prior_var  = float(prior_var)

    self._train = train
    self._validate = validate

    self._optimizers = []

    with tf.device("/cpu:0"):
      self.init_optimizers()

      input_tensors = tf.split(train.dequeue(), num_gpus, 0)

      # MultiGPU setup
      grads = []
      with tf.variable_scope(tf.get_variable_scope()):
        for i in range(num_gpus):
          with tf.name_scope('tower_%d' % i), tf.device('/gpu:%d' % i):
            losses = self._build_train_op(input_tensors[i])
            tf.get_variable_scope().reuse_variables()
            grads.append(self.compute_grads(losses))

      # Synchronize all towers, i.e. average their gradients.
      grad_steps = self.apply_grads(self.avg_grads(grads))
      self._train_op = tf.group(*grad_steps)
      self._loss = tf.reduce_mean(self._losses)

    # Build the rest of the stages
    self._build_eval_op()
    with tf.device('/cpu:0'):
      self._build_sample_op()
      self._build_reconstruct_op()

    # Check all properties set
    assert self._losses, "Loss operation not set"
    assert self._eval_op is not None, "Validation operation not set"
    assert self._sample_op is not None, "Sampling operation not set"

  def compute_grads(self, losses):
    assert len(losses) == len(self._optimizers)
    grads = []
    for loss, opt in zip(losses, self._optimizers):
      grads.append(opt.compute_gradients(loss))
    return grads

  def avg_grads(self, grads):
    grads_per_loss = zip(*grads)
    assert len(grads_per_loss[0]) == self._num_gpus
    assert len(grads_per_loss) == len(self._optimizers)

    avg = []
    for i in range(len(self._optimizers)):
      avg.append(gradients.avg(grads_per_loss[i]))

    return avg

  def apply_grads(self, grads):
    assert len(grads) == len(self._optimizers)

    grad_steps = []
    for grad, opt in zip(grads, self._optimizers):
      grad_steps.append(opt.apply_gradients(grad))

    return grad_steps

  # ----------------------------------------------------------------------------

  @property
  def init_op(self):
    return tf.global_variables_initializer()

  # ----------------------------------------------------------------------------

  def train(self, sess):
    return sess.run([ self._train_op, self._loss ])

  def reconstruct(self, sess):
    return sess.run(self._reconstr_op)

  def sample(self, sess):
    return sess.run(self._sample_op)

  def eval(self, sess):
    return sess.run([ self._eval_op ])

  # ----------------------------------------------------------------------------

  def _reconstruct(self, queue, prefix):
    """
    Helper method which takes a queue and a summary name and reconstructs
    an input image using the model.
    
    :param queue: A queue with batches of images.
    :param prefix: A summary name prefix.
    :return: The summary constructed.
    """
    x = queue.dequeue()

    rec_x, loss, rec_loss, lat_loss = self._model(x, True)

    x = tf.slice(x, [0, 0], [self._num_samples, -1])
    rec_x = tf.slice(rec_x, [0, 0], [self._num_samples, -1])

    samples_shape = (self._num_samples, self._input_size * 2, self._input_size, 1)
    x_rec_x = tf.reshape(tf.concat([x, rec_x], 1), samples_shape)

    latls = tf.summary.scalar(prefix + " " + const.LAT_LOSS, tf.reduce_mean(lat_loss))
    recls = tf.summary.scalar(prefix + " " + const.REC_LOSS, tf.reduce_mean(rec_loss))
    hists = tf.summary.histogram(prefix + const.REC_HIST, tf.divide(rec_loss, self._img_size))
    recos = tf.summary.image(prefix + " reconstruction", x_rec_x)

    return [recos, hists, latls, recls]

  # ----------------------------------------------------------------------------

  def _build_reconstruct_op(self):
    """
    Builds a reconstruction operation for both training and validation batches.
    """
    train_summ = self._reconstruct(self._train, "Training")
    val_summ = self._reconstruct(self._validate, "Validation")

    self._reconstr_op = train_summ + val_summ

  # ----------------------------------------------------------------------------

  def _build_eval_op(self):
    """
    Builds a LOSS summary for a validation batch.
    """

    self._eval_op = self._model(self._validate.dequeue(), True)[1]

  # ----------------------------------------------------------------------------

  def _build_sample_op(self):
    """
    Builds summaries for sampling the prior of the model. It also creates summaries
    to recycle the z-samples through the model several times, with the
    aim to get it closer to a denser area of the latent space.
    """

    self._build_prior_sample_op()

    assert (self._samples is not None), "Prior samples not set by _build_single_sample_op"

    num_samples = self._num_samples
    samples_shape = (num_samples, self._input_size, self._input_size, 1)
    self._sample_op = []

    x = self._samples
    imgs = tf.reshape(x, samples_shape)
    summary = tf.summary.image(const.PRIOR_SAMPLES, imgs, max_outputs=num_samples)
    self._sample_op.append(summary)

    for i in range(1, 6):
      x, _, _, _ = self._model(x, True)
      imgs = tf.reshape(x, samples_shape)
      summary = tf.summary.image(const.PRIOR_SAMPLES + "%d cycles" % i, imgs, max_outputs=num_samples)
      self._sample_op.append(summary)

  # ----------------------------------------------------------------------------

  @abstractmethod
  def _build_train_op(self, x): pass

  @abstractmethod
  def init_optimizers(self): pass

  @abstractmethod
  def _model(self, x, reuse): pass

  @abstractmethod
  def _build_prior_sample_op(self): pass

  @abstractmethod
  def get_runname(self): pass
