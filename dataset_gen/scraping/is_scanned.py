import subprocess
import sys
import glob
import httplib
import os

BASE_API_URL = "146.169.47.72:5000"

def get_output(command):
  process = subprocess.Popen(command, stdout=subprocess.PIPE)
  output  = process.communicate()[0]

  return str(output)[2:]

def pdf_info(pdffile):
  """
  Retrieves important information about a pdf file.
  ** Keys of interest: producer, pages
  """
  pdfinfo = get_output(["pdfinfo", pdffile]).split("\n")
  
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

  if "producer" in info and "abbyy" in info['producer'].lower():
    return True # OCR tool detected

  pages = int(info['pages'])

  images = get_output(["pdfimages", "-list", pdffile]).split("\n")[2:-1]
  imagecount = len(images)

  if (imagecount >= pages * 0.8): # TODO: add a more robust check
    return True

  return False

def main():
  assert len(sys.argv) == 2

  path = sys.argv[1]

  conn = httplib.HTTPConnection(BASE_API_URL)

  for pdffile in glob.glob(os.path.join(path, "*", "*.pdf")):
    score_id = os.path.splitext(os.path.basename(pdffile))[0]

    fullpath = os.path.join(path, score_id, pdffile)
    scanned = check_is_scanned(fullpath)
    
    conn.request("GET", "/info/" + score_id + ("?scanned=%d" % scanned))
    r = conn.getresponse()
    
    if not scanned:
      print(pdffile, scanned, r.status)

if __name__ == "__main__":
  main()
