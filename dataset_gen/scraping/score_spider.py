from urllib.parse import urljoin
import scrapy
from scrapy.http import Request, FormRequest
from scrapy.contrib.spiders.init import InitSpider
from scrapy.contrib.spiders import Rule
import os
import json
import time
import subprocess

BASE_API_URL = "http://146.169.46.120:5000"
BASE_IMSLP_URL = "http://imslp.org"
ACCEPT_DISC_URL = 'http://imslp.org/wiki/Special:IMSLPDisclaimerAccept/'

class ScoreSpider(InitSpider):
  name = 'ScoreSpider'

  login_page = 'http://imslp.org/wiki/Special:UserLogin'

  start_urls = [urljoin(BASE_API_URL, "poll")]

  username = 'George.krasenov'
  password = '0Le3NFenDO'

  SAVE_PATH = '/vol/bitbucket/gki13/dgmmn/piano'
  # SAVE_PATH = '.'
  PDF_LIMIT = 3

  def init_request(self):
    """This function is called before crawling starts."""
    return Request(url=self.login_page, callback=self.login)

  def login(self, response):
    """Generate a login request."""
    return FormRequest.from_response(
            response,
            formxpath='//*[@id="userloginForm"]/form',
            formdata={
              'wpName': self.username,
              'wpPassword': self.password
            },
            callback=self.check_login_response
          )

  def check_login_response(self, response):
    """Check the response returned by a login request to see if we are
    successfully logged in.
    """
    if "Main_Page" in response.url:
      self.log("Successfully logged in. Let's start crawling!")
      # Now the crawling can begin..

      for url in self.start_urls:
        yield self.make_requests_from_url(url)

      self.initialized()
    else:
      self.log("Logging in unsuccessful.")
      # Something went wrong, we couldn't log in, so nothing happens.

  def parse(self, response):
    resp = json.loads(response.body_as_unicode())
    
    score_id = resp['score_id']

    yield Request(urljoin(ACCEPT_DISC_URL, str(score_id)),
                  callback=lambda r, i=score_id: self.save_pdf(r, score_id))

  def count_bars(self, path):
    pages = os.listdir(path)

    pages_count = 0
    bars_count  = 0
    for p in pages:
      if p.startswith("_"):
        pages_count += 1

        for l in os.listdir(os.path.join(path, p)):
          bars_count += len(os.listdir(os.path.join(path, p, l)))

    return (pages_count, bars_count)

  def verify_complete(self, response):
    if response.status == 200:
      self.log("Complete successful.")
    else:
      self.log("Complete unsuccessful.")

  def save_pdf(self, response, score_id):
    if "PDF" in str(response.body[1:10]):
      self.log("PDF successfully retrieved.")
      pdfname = str(score_id) + ".pdf"
      path = os.path.join(self.SAVE_PATH, str(score_id))

      if not os.path.exists(path):
        os.makedirs(path)

      fpath = os.path.join(path, pdfname)

      with open(fpath, "wb") as f:
        f.write(response.body)

      ret_code = subprocess.call("python3 process.py %s" % fpath, shell=True)

      if ret_code != -1:
        (pages, bars) = self.count_bars(os.path.join(path, str(score_id)))

        yield Request(
            BASE_API_URL + "/complete/" + 
            str(score_id) + "?bars=%d&pages=%d&scanned=%d" % (bars, pages, ret_code),
            callback=self.verify_complete
        )
        self.log("Successfully retrieved PDF with %d pages, %d bars." % (pages, bars))
    else:
      self.log("Error Retrieving PDF.")

    time.sleep(5)

    yield Request(self.start_urls[0], callback=self.parse, dont_filter=True)
