#!/usr/local/bin/python3

import cv2
import sys
import os
import shutil
import numpy as np

from . import hough_splitting as hough
from . import exact_splitting as exact
from .util.imgcat import mat_cat

IMG_DENSITY_THRESH = 100
MIN_BAR_WIDTH      = 50
BAR_SIZE = (200, 200)

def main():
  outputPath = sys.argv[1]
  imagePath  = sys.argv[2]
  is_scanned = bool(int(sys.argv[3]))

  split((outputPath, imagePath, is_scanned))

def process_one_line(splitting, path, idx, line, bnwline):
  # Split bars
  bars = splitting.bar_split(line, bnwline)
  
  if bars is None: return 0

  line_path = os.path.join(path, "line_" + str(idx))
  if not os.path.exists(line_path):
    os.makedirs(line_path)

  bar_idx = 0
  for bar in bars:
    # Get rid of mostly empty bars, e.g. start and end sections
    horProj = cv2.reduce(bar, 1, cv2.REDUCE_AVG)
    proj = cv2.reduce(horProj, 0, cv2.REDUCE_MAX)

    if proj[0][0] < IMG_DENSITY_THRESH or bar.shape[1] < MIN_BAR_WIDTH:
      continue

    # Save the rest to the approriate locations
    bar_path = os.path.join(line_path, "bar_" + str(bar_idx) + ".png")
    resized = cv2.resize(bar, BAR_SIZE, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(bar_path, resized)
    bar_idx += 1

  # Remove empty line dirs
  if bar_idx == 0:
    shutil.rmtree(line_path)
    return 0

  return 1

def process_lines(splitting, path, lines, bnwlines, depth=0, idx=0):
  if depth > 3: return # limit recursion

  # Process lines
  line_idx = idx
  for line, bnwline in zip(lines, bnwlines):
    # Skip mostly empty lines
    vertProj = cv2.reduce(bnwline, 0, cv2.REDUCE_MAX)
    proj = cv2.reduce(vertProj, 1, cv2.REDUCE_AVG)
    if proj[0][0] < IMG_DENSITY_THRESH: continue

    if line.shape[0] > 500:
      smlines, smbnwlines = splitting.line_split(line, 8)
      # print(len(smbnwlines))
      if len(smlines) > 1:
        added = process_lines(splitting, path, smlines, smbnwlines, depth+1, line_idx)
        line_idx += abs(line_idx - added)
    else:
      line_idx += process_one_line(splitting, path, line_idx, line, bnwline)
  
  return line_idx

def split(args):
  # Unpack arguments
  path, imagePath, is_scanned = args

  # Choose splitting module
  splitting = hough if is_scanned else exact

  # Load image
  image = cv2.imread(imagePath)
  if image is None:
    raise Exception("Cannot load image '%s'." % imagePath)

  # Convert the image to Black (Background) & White (Text)
  kernel = np.ones((7,7), np.uint8)
  gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 150, apertureSize=3)
  edges = cv2.dilate(edges, kernel, iterations=3)

  vertProj = cv2.reduce(edges, 0, cv2.REDUCE_AVG)
  imgproj = cv2.reduce(vertProj, 1, cv2.REDUCE_AVG)[0][0]

  # Split lines
  lines, bnwlines = splitting.line_split(image)

  count = process_lines(splitting, path, lines, bnwlines)

  return imgproj

if __name__ == "__main__":
  main()