from __future__ import print_function

import cv2
import numpy as np

from .util.lineiter import get_line_iter, find_most_consecutive
from . import line_splitting as ls

MAX_INTENSITY = 255

# Hyperparameters
MIN_OVERLAP_FACTOR  = 0.5
SPACE_HEIGHT_THRESH = 8 # Ignore separating "spaces" with fewer rows 
LINE_DENSITY_THRESH = 0
COL_SKIP_THRESH     = MAX_INTENSITY * (MIN_OVERLAP_FACTOR**2)
LINE_HEIGHT_THRESH  = 0

def line_split(img, density_thresh=LINE_DENSITY_THRESH):
  return ls.line_split(img, SPACE_HEIGHT_THRESH, density_thresh, LINE_HEIGHT_THRESH)

def bar_split(line, bnwline):
  lrows, lcols, _ = line.shape

  # Compute thresholds
  overlap_thresh  = lrows * MIN_OVERLAP_FACTOR

  # Get the vertical project, i.e. avg intensity for every column.
  vertProj = cv2.reduce(bnwline, 0, cv2.REDUCE_AVG)[0]

  # Find the bar lines
  xcoords = []
  inBar = False
  for col in range(lcols):
    # Skip mostly empty columns
    if vertProj[col] < COL_SKIP_THRESH: continue

    it = [(0, 0, bnwline[i, col]) for i in range(lrows)]

    if find_most_consecutive(it) >= overlap_thresh:
      xcoords.append(col)
    else:
      inBar = False

  # Split into bars
  bars = []
  prev = 0
  for xcoord in xcoords:
    xcoord += 1
    bars.append(bnwline[:, prev:xcoord])
    prev = xcoord

  return bars
