import shutil
import os

def select(images, projections):

  threshold = 0.65 * max(projections)

  for (path, imagepath, _), proj in zip(images, projections):
    if proj < threshold and os.path.isdir(path):
      shutil.rmtree(path)
