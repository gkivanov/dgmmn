from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from tqdm import tqdm

import tensorflow as tf
from dgmmn import DGMMN

LR = 1e-5
LATENT_DIM = 1024
EPOCHS = 30
KEEP_PROB = 0.5


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying quality of music bars.
  Args:
    x: an input tensor with the dimensions (N_examples, 40000), where 40000 is the
    number of pixels in a standard DGMMN image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 2), with values
    equal to the logits of classifying the music bar into one of 2 classes (good 
    or bad). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  x_image = tf.reshape(x, [-1, 200, 200, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  w_conv1 = weight_variable("w_conv1", [5, 5, 1, 32])
  b_conv1 = bias_variable("b_conv1", [32])
  h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  w_conv2 = weight_variable("w_conv2", [5, 5, 32, 64])
  b_conv2 = bias_variable("b_conv2", [64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 200x200 image
  # is down to 50x50x64 feature maps -- maps this to 768 features.
  w_fc1 = weight_variable("w_fc1", [50 * 50 * 64, LATENT_DIM])
  b_fc1 = bias_variable("b_fc1", [LATENT_DIM])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 50*50*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

  # Dropout - controls the complexity of the model
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 768 features to 2 classes, for good and bad quality
  w_fc2 = weight_variable("w_fc2", [LATENT_DIM, 2])
  b_fc2 = bias_variable("b_fc2", [2])

  y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, w):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(name, shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)


def bias_variable(name, shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)


def main(_):
  # Import data
  # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  dgmmn = DGMMN(FLAGS.data_dir, 200)

  IMGSC = dgmmn.total_images

  # Initialize logs directory
  runname = "DISC_%d_%d_%.0E_%d_%g" % (IMGSC, EPOCHS, LR, LATENT_DIM, KEEP_PROB)
  idx = -1
  while True:
    idx += 1
    logs_dir = os.path.join(FLAGS.logs_dir, runname + ("_%d" % idx))
    if not tf.gfile.Exists(logs_dir): break

  tf.gfile.MakeDirs(logs_dir)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 40000])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  saver = tf.train.Saver()
  writer = tf.summary.FileWriter(logs_dir, graph=tf.get_default_graph())

  # Accuracy placeholders
  tracc_ph = tf.placeholder(tf.float32)
  valacc_ph = tf.placeholder(tf.float32)

  # Dataset epoch accuracy summaries
  tracc_s = tf.summary.scalar("Training Accuracy", tracc_ph)
  valacc_s = tf.summary.scalar("Validation Accuracy", valacc_ph)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("Starting training of model '%s'" % runname)

    for e in range(EPOCHS):
      ########################## TRAIN ##########################
      tracc = 0.0
      for _ in tqdm(range(dgmmn.train_batches_per_epoch)):
        # Get batch
        imgs, lbls = dgmmn.next_train_batch()
        fd = {x: imgs, y_: lbls, keep_prob: 1.0}

        # Evaluate
        tracc += accuracy.eval(feed_dict=fd)

        # Train
        fd[keep_prob] = KEEP_PROB
        train_step.run(feed_dict=fd)

      tracc /= dgmmn.train_batches_per_epoch
      print('epoch %d, training accuracy %g' % (e, tracc))

      summ = sess.run(tracc_s, feed_dict={tracc_ph: tracc})
      writer.add_summary(summ, e)

      ######################## VALIDATE #########################
      valacc = 0.0
      for _ in range(dgmmn.val_batches_per_epoch):
        # Get batch
        imgs, lbls = dgmmn.next_validate_batch()

        # Evaluate
        fd = { x: imgs, y_: lbls, keep_prob: 1.0 }
        valacc += accuracy.eval(feed_dict=fd)

      valacc /= dgmmn.val_batches_per_epoch
      print('epoch %d, test accuracy %g' % (e, valacc))

      summ = sess.run(valacc_s, feed_dict={valacc_ph: valacc})
      writer.add_summary(summ, e)

      writer.flush()

    model_path = os.path.join(logs_dir, "model.ckpt")
    saver.save(sess, model_path)
    print("Done. Model saved to '%s'" % model_path)

if __name__ == '__main__':
  flags = tf.flags

  flags.DEFINE_string("logs_dir", "/home/george/dgmmn/disclogs", "logs location")
  flags.DEFINE_string("data_dir", "/mnt/discdata", "dataset location")

  FLAGS = flags.FLAGS

  tf.app.run(main=main, argv=[sys.argv[0]])
