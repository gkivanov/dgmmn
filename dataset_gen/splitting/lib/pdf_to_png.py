#!/usr/local/bin/python3

import subprocess
import os
import traceback
import sys
 
 
# Absolute path to Ghostscript executable here or command name if Ghostscript is
# in your PATH.
GHOSTSCRIPTCMD = "gs"
 
 
def usage_exit():
  sys.exit("Usage: %s png_resolution pdffile1 pdffile2 ..." %
      os.path.basename(sys.argv[0]))
 
 
def main():
  if not len(sys.argv) >= 3:
    usage_exit()
  try:
    resolution = int(sys.argv[1])
  except ValueError:
    usage_exit()
  for filepath in sys.argv[1:]:
    (name, ext) = os.path.splitext(filepath)
    if ext.lower().endswith("pdf"):
      print("*** Converting %s..." % filepath)
      gs_pdf_to_png(os.path.join(os.getcwd(), filepath), resolution)
 
 
def gs_pdf_to_png(pdffilepath, resolution, outputPath=None):
  if not os.path.isfile(pdffilepath):
    print("'%s' is not a file. Skip." % pdffilepath)
  pdffiledir = os.path.dirname(pdffilepath)
  pdfname, ext = os.path.splitext(pdffilepath)
  output = os.path.join(outputPath, "_%03d.png")
  
  if not os.path.exists(outputPath):
    os.makedirs(outputPath)

  try:    
    # Change the "-rXXX" option to set the PNG's resolution.
    # http://ghostscript.com/doc/current/Devices.htm#File_formats
    # For other commandline options see
    # http://ghostscript.com/doc/current/Use.htm#Options
    arglist = [GHOSTSCRIPTCMD,
              "-dBATCH",
              "-dNOPAUSE",
              "-sOutputFile=%s" % output,
              "-sDEVICE=png16m",
              "-r%s" % resolution,
              pdffilepath]
    print("Converting pdf to images using GhostScript...")
    sp = subprocess.Popen(
        args=arglist,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
  except OSError:
    sys.exit("Error executing Ghostscript ('%s'). Is it in your PATH?" %
             GHOSTSCRIPTCMD)            
  except:
    print("Error while running Ghostscript subprocess. Traceback:")
    print("Traceback:\n%s"%traceback.format_exc())

  stdout, stderr = sp.communicate()
  if stderr:
    print("Ghostscript stdout:\n'%s'" % stdout)
    print("Ghostscript stderr:\n'%s'" % stderr)
 
 
if __name__ == "__main__":
  main()