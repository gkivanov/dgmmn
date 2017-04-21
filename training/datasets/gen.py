import glob
import os
from random import shuffle
import shutil
import tqdm

val_size = 0.2

datadir = "/mnt/dataclean"
outdir  = "/mnt/datafinal_split" + str(val_size)

traindir = os.path.join(outdir, "train")
testdir = os.path.join(outdir, "test")

if not os.path.exists(traindir):
  os.makedirs(traindir)

if not os.path.exists(testdir):
  os.makedirs(testdir)

filelist = glob.glob(os.path.join(datadir, "*", "*", "*", "*"))
shuffle(filelist)

print(len(filelist))

validx = int(len(filelist) * (1 - val_size))

trainpths = filelist[:validx]
testpths = filelist[validx:]


def run(paths, to_dir):
  for p in tqdm.tqdm(paths):
    psimple = p[len(datadir) + 1:]
    scoretxt, pagetxt, linetxt, bartxt = psimple.split("/")
    score = int(scoretxt)
    page = int(pagetxt[1:])
    line = int(linetxt.split("_")[1])
    bar = int(bartxt.split("_")[1].split(".")[0])
    newname = "%d_%d_%d_%d.png" % (score, page, line, bar)
    shutil.copy(p, os.path.join(to_dir, newname))

run(trainpths, traindir)
run(testpths, testdir)
