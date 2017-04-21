from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
from tqdm import tqdm

# DATASETS
from datasets import dgmmn, mnist

# MODELS
from models.vae_linear import LinearVAE as LinVAE
from models.vae_fc import FullyConnectedVAE as FConVAE
from models.vae_conv import ConvolutionalVAE as ConvVAE
from models.vae_stn import STNFullyConnectedVAE as STNFConVAE
from models.draw import DRAW

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("logs_dir", "/home/george/dgmmn/logs", "")
flags.DEFINE_string("dataloc", "/mnt/data", "dataset location")
flags.DEFINE_string("working_directory", "tmp", "")

flags.DEFINE_string("dataset", "dgmmn", "dataset selection")
flags.DEFINE_string("model", "linear_vae", "model")
flags.DEFINE_string("prefix", "", "runname prefix")

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("max_epoch", 2000, "max epoch")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_integer("hidden_size", 15, "size of the hidden VAE unit")
flags.DEFINE_integer("samples", 5, "samples to generate")
flags.DEFINE_integer("numgpus", 1, "number of gpus to use")

flags.DEFINE_integer("prior_mean", 0, "Prior mean")
flags.DEFINE_integer("prior_var", 1, "Prior variance")

flags.DEFINE_string("restore", "", "Path to restore variables from")

FLAGS = flags.FLAGS


def init_dirs(model, ds):
  """
  Initializes all directories for logging and model storage.
  
  :param model: The model selected
  :param ds: The dataset selected
  
  :return: The logging directory
  """
  datadir = os.path.join(FLAGS.working_directory, FLAGS.dataset.upper())
  imgsdir = os.path.join(FLAGS.working_directory, 'generated')
  if not os.path.exists(datadir): os.makedirs(datadir)
  if not os.path.exists(imgsdir): os.makedirs(imgsdir)

  runname = FLAGS.prefix + model.get_runname() + "_" + ds.get_runname()

  idx = -1
  while True:
    idx += 1
    logsdir = os.path.join(FLAGS.logs_dir, runname + ("_%d" % idx))
    if not tf.gfile.Exists(logsdir): break

  tf.gfile.MakeDirs(logsdir)

  return logsdir


def add_summaries(writer, summaries, step):
  """
  Writes summaries to disk.
  
  :param writer: the writer to be used
  :param summaries: a list of summaries or a single summary
  :param step: the current step (an integer)
  """
  if type(summaries) == list:
    for s in summaries:
      writer.add_summary(s, step)
  else:
    writer.add_summary(summaries, step)


def train(sess, model, dataset, size):
  """
  Runs the training stage for a single dataset epoch
  
  :param sess: The session itself
  :param model: The model to be trained
  :param dataset: The dataset delivering batches
  :param size: The input dimension of the images
  
  :return: The average loss for the epoch
  """
  iters = dataset.train_batches_per_epoch
  trloss = sum([model.train(sess)[1] for _ in tqdm(range(iters))])
  return trloss / iters


def validate(sess, model, dataset, size):
  """
  Runs the validation stage for a single dataset epich
  
  :param sess: The session itself
  :param model: The model to be trained
  :param dataset: The dataset delivering batches
  :param size: The input dimension of the images
  
  :return: The average loss for the epoch
  """
  iters = dataset.val_batches_per_epoch
  valloss = sum([model.eval(sess)[0] for _ in tqdm(range(iters))])
  return valloss / iters


def main():
  batch_size = FLAGS.batch_size
  samples = FLAGS.samples
  dataloc = FLAGS.dataloc
  learn_rate = FLAGS.learning_rate
  hid_size = FLAGS.hidden_size
  work_dir = FLAGS.working_directory
  data_dir = os.path.join(work_dir, FLAGS.dataset.upper())

  model, size, data, ds = None, None, None, None

  ##############################################################################
  ############################# DATASET SELECTION ##############################
  ##############################################################################
  assert FLAGS.dataset in ['dgmmn', 'mnist']

  with tf.device("/cpu:0"):
    if FLAGS.dataset == 'dgmmn':  # DGMMN
      size = 200
      ds = dgmmn.DGMMN(dataloc, batch_size * FLAGS.numgpus)

    elif FLAGS.dataset == 'mnist':  # MNIST
      size = 28
      ds = mnist.MNIST(data_dir, batch_size * FLAGS.numgpus)

    # Get dataset queue
    train_q = ds.train_queue((size, size), threads=2, capacity=400)
    validate_q = ds.validate_queue((size, size), threads=2, capacity=150)

  ##############################################################################
  ############################# MODEL SELECTION ################################
  ##############################################################################
  assert FLAGS.model in ['linvae', 'fcvae', 'cvae', 'stnfcvae', 'draw']

  if FLAGS.model == 'linvae':  # VAE
    model = LinVAE(train_q, validate_q, size, hid_size, learn_rate,
                   samples, FLAGS.numgpus, FLAGS.prior_mean, FLAGS.prior_var)

  elif FLAGS.model == 'fcvae':
    model = FConVAE(train_q, validate_q, size, hid_size, learn_rate,
                    samples, FLAGS.numgpus, FLAGS.prior_mean, FLAGS.prior_var)

  elif FLAGS.model == 'cvae':
    model = ConvVAE(train_q, validate_q, size, hid_size, learn_rate,
                    samples, FLAGS.numgpus, FLAGS.prior_mean, FLAGS.prior_var)

  elif FLAGS.model == 'stnfcvae':
    model = STNFConVAE(train_q, validate_q, size, hid_size, learn_rate,
                       samples, FLAGS.numgpus, FLAGS.prior_mean, FLAGS.prior_var)

  elif FLAGS.model == 'draw':  # DRAW
    model = DRAW(train_q, validate_q, size, hid_size, learn_rate,
                 samples, FLAGS.numgpus, FLAGS.prior_mean, FLAGS.prior_var)

  ##############################################################################
  ############################### RUN SESSION ##################################
  ##############################################################################

  # Initialize Session
  sess = tf.InteractiveSession()
  sess.run(model.init_op)

  # Model Saver
  saver  = tf.train.Saver()

  # Restore model if path given
  if len(FLAGS.restore) > 0:
    saver.restore(sess, os.path.join(FLAGS.restore, "model.ckpt"))

  # TensorBoard Summary writer
  logs_dir = init_dirs(model, ds)
  writer = tf.summary.FileWriter(logs_dir, graph=tf.get_default_graph())

  # Initialize Queue Coordinator
  coord   = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  # Loss placeholders
  trloss_ph  = tf.placeholder(tf.float32)
  valloss_ph = tf.placeholder(tf.float32)

  # Dataset epoch loss summaries
  trloss_s  = tf.summary.scalar("Training Loss per ds epoch", trloss_ph)
  valloss_s = tf.summary.scalar("Validation Loss per ds epoch", valloss_ph)

  print("Training starting for model '%s'" % os.path.basename(logs_dir))

  try:
    counter = 1
    while counter <= FLAGS.max_epoch:

      # Training Stage
      trloss = train(sess, model, ds, size)
      fd = {trloss_ph: trloss}
      add_summaries(writer, sess.run(trloss_s, feed_dict=fd), counter)

      # Validation stage
      valloss = validate(sess, model, ds, size)
      fd = {valloss_ph: valloss}
      add_summaries(writer, sess.run(valloss_s, feed_dict=fd), counter)

      # Training and Validation Samples Reconstruction
      add_summaries(writer, model.reconstruct(sess), counter)

      # Prior sampling (generation)
      add_summaries(writer, model.sample(sess), counter)

      # Stdout progress update
      print("epochs=%d> TRLOSS: %f; VALLOSS: %f" % (counter, trloss, valloss))

      # Epochs counter
      counter += 1
  except Exception as e:
    coord.request_stop(e)
  finally:
    # Save model
    saver.save(sess, os.path.join(logs_dir, "model.ckpt"))
    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
  main()
