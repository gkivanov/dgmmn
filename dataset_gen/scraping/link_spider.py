from scrapy import signals
from scrapy.xlib.pydispatch import dispatcher
from urllib.parse import urljoin
import scrapy
from scrapy.http import Request, FormRequest
from scrapy.contrib.spiders.init import InitSpider
from scrapy.contrib.spiders import Rule
import os
import MySQLdb
import time
import re

class ScoreSpider(InitSpider):
  name = 'ScoreSpider'

  db  = MySQLdb.connect("localhost", "root", "", "dgmmn")

  ACCEPT_DISC_URL = '/wiki/Special:IMSLPDisclaimerAccept/'

  login_page = 'http://imslp.org/wiki/Special:UserLogin'

  start_urls = ['http://imslp.org/index.php?title=Category:For_piano']

  username = 'George.krasenov'
  password = '0Le3NFenDO'

  SAVE_PATH = '/vol/bitbucket/gki13/dgmmn/piano'
  PDF_LIMIT = 3

  BASE_URL = 'http://imslp.org/'

  count = 0

  def __init__(self):
    dispatcher.connect(self.quit, signals.spider_closed)

  def quit(self, spider):
    self.log("Closing db connection...")
    self.db.close()

  def init_request(self):
    """This function is called before crawling starts."""
    # return Request(url=self.login_page, callback=self.login)

    for url in self.start_urls:
      yield self.make_requests_from_url(url)

    self.initialized()

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
    yield Request(response.url, callback=self.single_page_links)

  def single_page_links(self, r):
    hrefs = r.selector.xpath('//*[@id="mw-pages"]/div[2]/table').css('a::attr(href)').extract()
    titles = r.selector.xpath('//*[@id="mw-pages"]/div[2]/table').css('a::attr(title)').extract()

    assert len(hrefs) == len(titles)

    for href, title in zip(hrefs, titles):
      yield Request(urljoin(r.url, href), callback=lambda r, t=title: self.parse_score_page(r, t))
      time.sleep(0.3)

    time.sleep(1)

    prev_next = r.selector.xpath('//*[@id="mw-pages"]/div[1]/a')
    nxt = prev_next[1] if len(prev_next) > 1 else prev_next[0]

    if "next" in nxt.css("a::text").extract_first():
      next_page_url = urljoin(r.url, nxt.css("a::attr(href)").extract_first())

      yield Request(next_page_url, callback=self.single_page_links)

  def parse_score_page(self, response, title):
    pdf_links = response.selector.xpath('//*[@class="we_file_first"]/div[1]/p/b').css('a::attr(href)').extract()
    score_ids = map(lambda l: l.split('/')[-1], pdf_links)

    rows_raw = response.selector.xpath('//*[@class="wi_body"]/table/tr')

    rows = {}
    for row in rows_raw:

      key = (row.css('th span::text') or row.css('th::text')).extract_first().replace('\n', '').lower()
      value = row.css('td')

      if key == 'lication' or 'year' in key:
        text = row.css("td::text").extract_first()
        m    = re.search("[12][0-9]{3}", text)
        
        if m is not None:
          rows["year"] = m.group(0)

      else:
        rows[key] = (row.css('td a::text') or row.css('td::text')).extract_first().replace('\n', '')

    year   = rows['year'] if 'year' in rows else 0
    author = rows['composer'] if 'composer' in rows else ''
    style  = rows['piece style'] if 'piece style' in rows else ''
    instr  = rows['instrumentation'] if 'instrumentation' in rows else ''

    # accept_url = urljoin(response.url, self.ACCEPT_DISC_URL + score_id)

    for sid in score_ids:

      cursor = self.db.cursor()

      sql = """INSERT INTO scores(href, score_id, author, title, year, style)
               VALUES ('%s', %d, '%s', '%s', %d, '%s')""" % (
                MySQLdb.escape_string(response.url).decode("utf-8", "ignore"),
                int(sid),
                MySQLdb.escape_string(author).decode("utf-8", "ignore"),
                MySQLdb.escape_string(title).decode("utf-8", "ignore"),
                int(year),
                MySQLdb.escape_string(style).decode("utf-8", "ignore"))

      try:
        cursor.execute(sql)
        self.db.commit()
        cursor.close()
        self.log("Successfully added score id: %d" % int(sid))
      except Exception as e:
        self.log("Error: %s" % str(e))
      self.db.rollback()
