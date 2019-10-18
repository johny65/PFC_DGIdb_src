"""
Web scraping para Springer.
"""
from html.parser import HTMLParser
import urllib.parse
import utils


class SpringerParser(HTMLParser):
    the_url = None
    process = True

    def __init__(self, url):
        HTMLParser.__init__(self)
        self.url = url
        data = utils.fetch_url(url)
        self.feed(data)

    def handle_starttag(self, tag, attrs):
        if self.process and tag == "a" and ("data-track-action", "Pdf download") in attrs:
            for attr in attrs:
                if attr[0] == "href":
                    self.the_url = utils.get_server(self.url) + urllib.parse.unquote(attr[1])
                    self.process = False

def scrap(url):
    parser = SpringerParser(url)
    return parser.the_url

if __name__ == "__main__":
    print(scrap("https://link.springer.com/article/10.1007/s00432-014-1589-3"))
