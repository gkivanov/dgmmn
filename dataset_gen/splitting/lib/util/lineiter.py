import numpy as np
import cv2

def get_line_iter(P1, P2, img):
  """
  Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

  Parameters:
      -P1: a numpy array that consists of the coordinate of the first point (x,y)
      -P2: a numpy array that consists of the coordinate of the second point (x,y)
      -img: the image being processed

  Returns:
      -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
  """
  imageH = img.shape[0]
  imageW = img.shape[1]
  P1X = P1[0]
  P1Y = P1[1]
  P2X = P2[0]
  P2Y = P2[1]

  # Difference and absolute difference between points
  # used to calculate slope and relative location between points
  dX = P2X - P1X
  dY = P2Y - P1Y
  dXa = np.abs(dX)
  dYa = np.abs(dY)

  # Predefine numpy array for output based on distance between points
  itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
  itbuffer.fill(np.nan)

  # Obtain coordinates along the line using a form of Bresenham's algorithm
  negY = P1Y > P2Y
  negX = P1X > P2X
  if P1X == P2X: # vertical line segment
     itbuffer[:, 0] = P1X
     if negY:
         itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
     else:
         itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
  elif P1Y == P2Y: # horizontal line segment
     itbuffer[:, 1] = P1Y
     if negX:
         itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
     else:
         itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
  else: # diagonal line segment
     steepSlope = dYa > dXa
     if steepSlope:
         slope = dX.astype(np.float32) / dY.astype(np.float32)
         if negY:
             itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
         else:
             itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
         itbuffer[:, 0] = (slope*(itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
     else:
         slope = dY.astype(np.float32) / dX.astype(np.float32)
         if negX:
             itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
         else:
             itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
         itbuffer[:, 1] = (slope*(itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

  # Remove points outside of image
  colX = itbuffer[:, 0]
  colY = itbuffer[:, 1]
  itbuffer = itbuffer[(colX>=0) & (colY>=0) & (colX<imageW) & (colY<imageH)]

  # Get intensities from img ndarray
  itbuffer[:, 2] = img[
    itbuffer[:, 1].astype(np.uint),
    itbuffer[:, 0].astype(np.uint)
  ]

  return itbuffer

def find_most_consecutive(it):
  """
  Returns the length of the longest *white* line in the image.
  NOTE: Works only on Black & White images.
  """
  inLine = False
  curr = 0
  longest = -1
  for _, _, intensity in it:

    if not inLine:
      if intensity > 0:
        inLine = True
        curr = 1
    else:
      if intensity > 0:
        curr += 1
      else:
        inLine = False
        longest = max(longest, curr)
        curr = 0

  if inLine:
    longest = max(longest, curr)

  return longest

def group_lines(lines, mean, img):
  """
  Groups lines into clusters based on their x1, x2 coordinates.
  """
  lines = sorted(lines, key=lambda line: line[0])
  prev = None
  group = []
  for line in lines:
    # Skip lines which have overlap smaller than the average for all lines.
    if line[4] < mean: continue

    # x1, y1, x2, y2, _ = line
    # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    if not prev or abs(line[0] - prev[0]) <= 25:
      group.append(line)
    else:
      yield group
      group = [line]
    prev = line

  # Yield the final group
  if group:
    yield group

