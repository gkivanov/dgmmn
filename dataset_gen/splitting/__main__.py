
import sys
from lib.split import split

def main():
  output = sys.argv[1]
  imgpath = sys.argv[2]
  is_scanned = bool(int(sys.argv[3]))

  split((output, imgpath, is_scanned))

if __name__ == "__main__":
  main()