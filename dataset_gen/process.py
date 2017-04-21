#!/usr/local/bin/python3

import sys
import os
from os.path import join as pjoin
from multiprocessing import Pool

from splitting.lib.pdf_to_png import gs_pdf_to_png
from splitting.lib.split      import split
from splitting.lib.select     import select
import subprocess
from pprint import pprint

RESOLUTION = 200
THREAD_POOL = 8

def get_output(command):
  process = subprocess.Popen(command, stdout=subprocess.PIPE)
  output  = process.communicate()[0]

  return str(output)[2:]

def pdf_info(pdffile):
  """
  Retrieves important information about a pdf file.
  ** Keys of interest: producer, pages
  """
  pdfinfo = get_output(["pdfinfo", pdffile]).split("\\n")
  
  info = {}

  for line in pdfinfo:
    tokenized = line.split(':', 1)
    if len(tokenized) != 2: continue
    info[str(tokenized[0].lower())] = tokenized[1].lstrip()

  return info

def check_is_scanned(pdffile):
  """
  Determines if a pdf file consists of scanned images or it is typefaced.
  """
  info = pdf_info(pdffile)

  print("============== PDF File information: ==============")
  pprint(info)
  print("===================================================")

  if "producer" in info and "abbyy" in info['producer'].lower():
    return True # OCR tool detected

  pages = int(info['pages'])

  images = get_output(["pdfimages", "-list", pdffile]).split("\\n")[2:-1]
  imagecount = len(images)

  if (imagecount >= pages * 0.8): # TODO: add a more robust check
    return True

  return False

def process(pdffile):
  if not os.path.isfile(pdffile):
    print("Cannot open file '%s'. Aborting..." % pdffile)
    raise Exception("Cannot open file!")

  is_scanned = check_is_scanned(pdffile)

  pdfname, ext = os.path.splitext(pdffile)
  pdfname = pdfname.replace(" ", "_").lower()
  output = os.path.join(pdfname, "raw")

  gs_pdf_to_png(pdffile, RESOLUTION, output)
  count = len(os.listdir(output))
  print("%d pages processed in '%s'. Scanned? %d" % (count, output, is_scanned))

  # Prepare arguments for splitting each page
  imgs = [
    (
      pjoin(pdfname, "_%03d" % i),
      pjoin(output, "_%03d.png" % i),
      is_scanned
    )
    for i in range(1, count + 1)
  ]

  # Split all pages in parallel
  with Pool(THREAD_POOL) as p:
    projections = p.map(split, imgs)

  # Sequential
  # projections = [split(img) for img in imgs]

  select(imgs, projections)

  print("All done.")
  return is_scanned

def main():
  pdffile = sys.argv[1]
  try:
    scanned = process(pdffile)
    sys.exit(int(scanned))
  except Exception as e:
    print(str(e))
    sys.exit(-1)

if __name__ == "__main__":
  main()