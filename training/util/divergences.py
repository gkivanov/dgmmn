import tensorflow as tf


def kl(mu_a, var_a, mu_b, var_b):
  """Calculate the batched KL divergence KL(n_a || n_b) with n_a and n_b Normal.

  Returns:
    Batchwise KL(n_a || n_b)
  """
  one = tf.constant(1, dtype=tf.float32)
  two = tf.constant(2, dtype=tf.float32)
  half = tf.constant(0.5, dtype=tf.float32)
  ratio = var_a / var_b
  return (tf.square(mu_a - mu_b) / (two * var_b) +
          half * (ratio - one - tf.log(ratio)))
