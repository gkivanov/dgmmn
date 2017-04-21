from __future__ import print_function

import cv2
import numpy as np

COL_DISCARD_THRESH  = 10 # intensity threshold
SPACE_HEIGHT_THRESH = 10 # Ignore separating "spaces" with fewer rows 
LINE_DENSITY_THRESH = 3
LINE_HEIGHT_THRESH  = 300
IMG_DENSITY_THRESH  = 50

def line_split(img,
               space_height_thresh=SPACE_HEIGHT_THRESH,
               density_thresh=LINE_DENSITY_THRESH,
               line_height_thresh=LINE_HEIGHT_THRESH):
  # Load the image
  rows, cols, _ = img.shape

  # Convert the image to Black (Background) & White (Text)
  gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
  bnw       = cv2.bitwise_and(gray, gray, mask=mask)
  ret, bnw  = cv2.threshold(gray, 180 , 255, cv2.THRESH_BINARY_INV)

  # TODO: Add orientiation fix
  # pts = cv2.findNonZero(bnw)

  # Get horizontal projections, i.e. average intensity for every line
  horProj = cv2.reduce(bnw, 1, cv2.REDUCE_AVG)
  proj = cv2.reduce(horProj, 0, cv2.REDUCE_MAX)

  if proj[0][0] < IMG_DENSITY_THRESH:
    return [], []

  blankSpaces = []
  start = 0
  count = 0
  isSpace = False
  initial = True

  # Find separating lines
  for cur in range(rows):
    isEmptyRow = horProj[cur] <= density_thresh

    if not isSpace:
      if isEmptyRow:
        if initial or (len(blankSpaces) > 0 and (cur - blankSpaces[-1][1]) >= line_height_thresh):
          initial = False
          isSpace = True
          count = 1
          start = cur
    else:
      if not isEmptyRow:
        isSpace = False
        if count >= space_height_thresh:
          blankSpaces.append((start, cur - space_height_thresh))
      else:
        count += 1

  if isSpace:
    blankSpaces.append((start, rows))

  # Extract Notation Lines
  lines = []
  bnwlines = []
  for i in range(len(blankSpaces)):
    prev = blankSpaces[i]
    start = prev[1]

    if (i + 1) >= len(blankSpaces):
      break
    else:
      curr = blankSpaces[i + 1]
      end  = curr[0]

    lines.append(img[start:end])
    bnwlines.append(bnw[start:end])

  return lines, bnwlines
