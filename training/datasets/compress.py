import glob
import os
import tqdm
import numpy as np
from PIL import Image

val_size = 0.2

datadir = "/mnt/datafinal_split0.1/test"
outdir  = datadir + "bins"

if not os.path.exists(outdir):
  os.makedirs(outdir)

filelist = glob.glob(os.path.join(datadir, "*"))

print(len(filelist))

binidx = 0
imgidx = 0

IMGSIZE_L = 40001
IMGS_PER_BIN = 25000

stacked = np.empty((IMGS_PER_BIN, IMGSIZE_L))
for p in tqdm.tqdm(filelist):
  imgname = os.path.basename(p)
  score, page, line, bar = map(int, imgname[:-4].split("_"))
  label = [line * 10 + bar]
  im = np.array(Image.open(p)).flatten()
  iml = np.array(list(label) + list(im), np.uint8)
  stacked[imgidx] = iml
  imgidx += 1

  if imgidx >= IMGS_PER_BIN:
    print("Saving to file...")
    stacked.tofile(os.path.join(outdir, "batch_%d.bin" % binidx))
    binidx += 1
    imgidx = 0
    stacked = np.empty((IMGS_PER_BIN, IMGSIZE_L))

print("Last one:")
print("Saving to file...")
stacked.tofile(os.path.join(outdir, "batch_%d.bin" % binidx))
