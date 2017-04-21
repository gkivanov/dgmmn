from flask import Flask, jsonify, request, send_file
import glob
import os
import base64
import shutil
import itertools
from nscanned import NSCANNED

LOADED = [
  107268,  169623,  222994,  254080,  272836,  335319,  354593,  371568,  381262,  431243,  7955,   84459,
  109666,  173071,  249345,  267164,  277360,  335967,  354668,  380618,  382074,  434276,  81868,  94587,
  133471,  181766,  251025,  268495,  290381,  336247,  359743,  380924,  397849,  449469,  82656,
  159993,  201697,  251026,  269959,  305766,  349980,  366291,  381041,  401212,  65656,  82703,
]

# for i in LOADED:
#   NSCANNED.remove(i)

app = Flask(__name__, static_url_path='/static')

datapath = "/mnt/datafull"
filelist = glob.glob(os.path.join(datapath, "*", "*", "*", "*"))

cleanpath = "/mnt/datafull/clean2"
dirtypath = "/mnt/datafull/dirty2"


def get_score_id(path):
  rest = path[(len(datapath) + 1):]
  tokens = rest.split("/")
  return int(tokens[0])


def get_file_gen():
  for f in filelist:
    yield f

file_gen = get_file_gen()


def get_id_for_dir(datadir):
  return len(os.listdir(datadir))


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route("/poll")
def poll():
  filepath = next(file_gen)
  while get_score_id(filepath) in NSCANNED:
    filepath = next(file_gen)

  with open(filepath, "rb") as f:
    encoded = base64.b64encode(f.read())
    return jsonify({"b64image": encoded, "filepath": filepath})


@app.route("/revert")
def revert():
  global file_gen

  savepath = request.args.get("savepath")
  imgpath  = request.args.get("imgpath")
  nxtimg   = request.args.get("nxtimg")

  os.remove(savepath)

  file_gen = itertools.chain([nxtimg], file_gen)

  with open(imgpath, "rb") as f:
    encoded = base64.b64encode(f.read())
    return jsonify({"b64image": encoded, "filepath": imgpath})


@app.route("/vote")
def vote():
  fname = request.args.get('fname')
  voter = int(request.args.get('vote'))

  if voter == 1:
    path = os.path.join(cleanpath, "%d.png" % get_id_for_dir(cleanpath))
  else:
    path = os.path.join(dirtypath, "%d.png" % get_id_for_dir(dirtypath))

  shutil.copy(fname, path)

  return jsonify({"path": path})

if __name__ == "__main__":
  app.run(host='0.0.0.0')
