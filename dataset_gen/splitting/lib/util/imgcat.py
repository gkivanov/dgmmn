import cv2
from subprocess import call

TMP_PATH = "/tmp/to_cat.png"

def mat_cat(data, width='auto', height='auto', preserveAspectRatio=False, inline=True, filename=''):
  """
  Saves the image to a temporary location, and CATs it with imgcat.
  After that the image is removed.

  NOTE: Works only in Unix based system due to the debugging nature of this
        method.
  """
  cv2.imwrite(TMP_PATH, data)
  call(["imgcat", TMP_PATH])
  call(["rm", TMP_PATH])
