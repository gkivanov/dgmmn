import cv2
import numpy as np
import math

from . import line_splitting as ls
from .util.lineiter import get_line_iter, find_most_consecutive, group_lines
from .util.imgcat import mat_cat

# Hyperparameters
MIN_LINE_LENGTH_FACTOR = 0.4
MIN_OVERLAP_THRESH     = 100 # The higher, the stricter we are about bar lines
LINE_HEIGHT_THRESH     = 5 # Ignore separating "spaces" with fewer rows 
LINE_DENSITY_THRESH    = 2 # The smaller, the stricter we are about hor lines
DILATION_ITERATIONS    = 3 # The higher, the more dilated the lines are
DILATION_KERNEL_SIZE   = (3, 3) # The bigger, the more dilated the lines are
DILATION_KERNEL        = np.ones(DILATION_KERNEL_SIZE, np.uint8)
# DILATION_KERNEL = np.array(
#   [[0,0,1,0,0],
#   [0,1,1,1,0],
#   [0,1,1,1,0],
#   [0,1,1,1,0],
#   [0,0,1,0,0]],
# dtype=np.uint8)

def get_coords(rho, theta):
  a  = np.cos(theta)
  b  = np.sin(theta)
  x0 = a * rho
  y0 = b * rho
  x1 = int(x0 + 1000*(-b))
  y1 = int(y0 + 1000*(a))
  x2 = int(x0 - 1000*(-b))
  y2 = int(y0 - 1000*(a))

  return x1, y1, x2, y2

def line_split(img, density_thresh=LINE_DENSITY_THRESH):
  return ls.line_split(img, LINE_HEIGHT_THRESH, density_thresh)

def bar_split(img, bnwline):
  rows, cols, _ = img.shape

  gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 150, apertureSize=3)
  edges = cv2.dilate(edges, DILATION_KERNEL, iterations=DILATION_ITERATIONS)

  # mat_cat(edges)

  min_line_length = MIN_LINE_LENGTH_FACTOR * rows

  # Detect Hough Lines
  houghlines = cv2.HoughLines(edges, 0.85, np.pi / 360, 210)
  if houghlines is None:
    return []

  # Process all found lines
  overlaps = []
  for line in houghlines:
    rho, theta = line[0]

    # Calculate line angle
    angle = abs(theta * 180 / np.pi)

    # Keep only vertical lines
    if angle >= 2.5 and angle < 177.5: continue

    x1, y1, x2, y2 = get_coords(rho, theta)

    # Align all lines in the same direction
    if angle < 90: 
      y1, y2 = -y1, -y2

    P1, P2 = np.array((x1, y1)), np.array((x2, y2))
    longest = find_most_consecutive(get_line_iter(P1, P2, edges))

    if (longest != -1 and longest > MIN_OVERLAP_THRESH):
      overlaps.append((x1, y1, x2, y2, longest))

  # If no bar line candidates found, return
  if len(overlaps) == 0:
    return []

  # Compute average overlap
  mean = np.mean([ov for _, _, _, _, ov in overlaps])

  bars = []
  prev = 0
  for group in group_lines(overlaps, mean, img):
    x1, y1, x2, y2, _ = map(int, np.mean(group, axis=0))
    
    # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # candidate

    P1, P2  = np.array((x1, y1)), np.array((x2, y2))
    longest = find_most_consecutive(get_line_iter(P1, P2, edges))

    if longest < min_line_length: continue

    x = int((x1 + x2) / 2)
    bars.append(bnwline[:, prev:x])
    prev = x

    # cv2.line(img, (x, 0), (x, rows), (0, 255, 0), 2) # bar line

  bars.append(bnwline[:, prev:cols])

  return bars
