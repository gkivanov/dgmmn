from __future__ import absolute_import, division, print_function

import tensorflow as tf
from train import deepnn
import os
import numpy as np
from scipy.misc import imread
import glob
import math
from tqdm import tqdm
import shutil

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("save_dir", "/mnt/dataclean", "clean dataset location")
flags.DEFINE_string("data_dir", "/mnt/datafull", "dataset location")

FLAGS = flags.FLAGS

MODEL_DIR = "/home/george/dgmmn/disclogs/DISC_10754_30_1E-05_1024_0_1"
BATCH_SIZE = 100

# Create the model
x = tf.placeholder(tf.float32, [None, 40000])

# Build the graph for the deep net
y_conv, keep_prob = deepnn(x)
prediction_op = tf.argmax(y_conv, 1)


def reformat(imgs):
  # Convert shape from [num examples, rows, columns, depth]
  # to [num examples, rows*columns] (assuming depth == 1)
  assert imgs.shape[3] == 1
  imgs = imgs.reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2])

  # Convert from [0, 255] -> [0.0, 1.0].
  imgs = imgs.astype(np.float32)
  imgs = np.multiply(imgs, 1.0 / 255.0)

  return imgs


def get_session(model_dir=MODEL_DIR):
  saver = tf.train.Saver()
  sess = tf.InteractiveSession()
  saver.restore(sess, os.path.join(model_dir, "model.ckpt"))
  return sess


def classify(sess, img_paths):
  imgs = [imread(p) for p in img_paths]
  imgs = reformat(np.expand_dims(np.array(imgs), axis=3))

  fd = { x: imgs, keep_prob: 1.0 }
  classification = sess.run(prediction_op, feed_dict=fd)

  return classification


def main():
  data_dir = FLAGS.data_dir
  filelist = glob.glob(os.path.join(data_dir, "*", "*", "*", "*"))

  batch_count = int(math.ceil(len(filelist) / BATCH_SIZE))

  sess = get_session()

  try:
    for bidx in tqdm(range(batch_count)):
      if bidx == 0:
        imgpaths = filelist[:BATCH_SIZE]
      else:
        imgpaths = filelist[(bidx * BATCH_SIZE):((bidx + 1) * BATCH_SIZE)]

      classification = classify(sess, imgpaths)

      for i in range(len(imgpaths)):
        if classification[i] == 0: continue

        path = imgpaths[i]

        basename = os.path.basename(path)

        dirname = os.path.dirname(path)
        dirname = dirname[(len(FLAGS.data_dir) + 1):]
        dirname = os.path.join(FLAGS.save_dir, dirname)

        if not tf.gfile.Exists(dirname):
          tf.gfile.MakeDirs(dirname)

        shutil.copy(path, os.path.join(dirname, basename))

  finally:
    sess.close()

if __name__ == "__main__":
  main()
