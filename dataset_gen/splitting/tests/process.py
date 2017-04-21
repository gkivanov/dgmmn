import unittest
import os
import shutil
from multiprocessing import Pool

from ..lib.split  import split
from ..lib.select import select

THREAD_POOL = 4

BASE_PATH      = os.path.join("splitting", "tests")
TMP_PATH       = os.path.join(BASE_PATH, "tmp")

FANTAISIE_PATH = os.path.join(BASE_PATH, "resources", "fantaisie")
POLKA_PATH     = os.path.join(BASE_PATH, "resources", "five_pound_polka_the")
PRINTEMPS_PATH = os.path.join(BASE_PATH, "resources", "printemps")

FANTAISIE_SPEC = [
  [5, 4, 4, 4, 4],
  [4, 4, 4, 4, 4],
  [4, 4, 3, 4, 4],
  [4, 3, 3, 3, 3],
  [3, 4, 3, 4, 3],
  [4, 4, 4, 3, 4],
  [4, 4, 3, 3, 3],
  [3, 4, 4, 4, 4],
  [4, 4, 4, 4, 5]
]

POLKA_SPEC = [
  [6, 5, 6, 5],
  [6, 5, 5, 5, 6],
  [6, 6, 6, 6, 6],
  [6, 5, 6, 5, 7],
  [6, 6, 6, 5, 6],
  [6, 5, 5, 5, 6]
]

PRINTEMPS_SPEC = [
  [3, 3, 4, 2],
  [4, 4, 3, 4, 5],
  [3, 3, 3, 3, 3],
  [3, 3, 3, 3, 3],
  [4, 4, 4, 3, 3],
  [2, 4, 4, 3, 6],
  [2, 2, 2, 2, 3], # 3 should be 2
  [2, 2, 2, 2, 2],
  [2, 3, 2, 3, 2], # all 3s should be 2
  [2, 2, 2, 2, 2]
]

class TestProcess(unittest.TestCase):

  def setUp(self):
    if os.path.exists(TMP_PATH):
      shutil.rmtree(TMP_PATH)

    os.makedirs(TMP_PATH)

  def check_spec(self, spec, path):
    pages = os.listdir(path)

    self.assertEqual(len(pages), len(spec), "Expected %d pages, got %d." % (len(spec), len(pages)))

    for page_idx, page in enumerate(pages):
      pagedir = os.path.join(path, page)
      lines = os.listdir(pagedir)

      actual_lines   = len(lines)
      expected_lines = len(spec[page_idx])

      self.assertEqual(actual_lines, expected_lines,
        "Expected %d lines for page %d, but got %d." % (expected_lines, page_idx, actual_lines))

      for line_idx, line in enumerate(lines):
        linedir = os.path.join(pagedir, line)
        bars    = os.listdir(linedir)

        actual_bars = len(bars)
        expected_bars = spec[page_idx][line_idx]

        self.assertEqual(actual_bars, expected_bars,
          "Expected %d bars for (P%d, L%d), but got %d." % (expected_bars, page_idx, line_idx, actual_bars))

  def verify_score(self, path, name, spec, is_scanned):
    imgs = os.listdir(path)
    output_path = os.path.join(TMP_PATH, name)

    # Prepare arguments for splitting each page
    imgs = [
      (
        os.path.join(output_path, os.path.splitext(img)[0]),
        os.path.join(path, img),
        is_scanned
      )
      for img in imgs
    ]

    # Split all pages in parallel
    with Pool(THREAD_POOL) as p:
      projections = p.map(split, imgs)

    select(imgs, projections)

    self.check_spec(spec, output_path)

  def test_fantaisie(self):
    self.verify_score(FANTAISIE_PATH, "fantaisie", FANTAISIE_SPEC, True)

  def test_polka(self):
    self.verify_score(POLKA_PATH, "polka", POLKA_SPEC, False)

  def test_printemps(self):
    self.verify_score(PRINTEMPS_PATH, "printemps", PRINTEMPS_SPEC, True)

if __name__ == '__main__':
  unittest.main()