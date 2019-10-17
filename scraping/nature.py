"""
Web scraping para Nature.com.
"""
from html.parser import HTMLParser
import urllib.parse
import utils


class NatureParser(HTMLParser):
    the_url = None
    a_attrs = None
    in_span = False

    def __init__(self, url):
        HTMLParser.__init__(self)
        self.url = url
        data = utils.fetch_url(url)
        self.feed(data)

    def handle_starttag(self, tag, attrs):
        self.in_span = (tag == "span")
        if tag == "a":
            self.a_attrs = attrs

    def handle_data(self, data):
        if self.in_span and data == "Download PDF":
            for attr in self.a_attrs:
                if attr[0] == "href":
                    self.the_url = utils.get_base_server(self.url) + urllib.parse.unquote(attr[1])

def scrap(url):
    parser = NatureParser(url)
    return parser.the_url

if __name__ == "__main__":
    print(scrap("https://www.nature.com/articles/pr197899?code=07b62691-c307-4345-9a58-932fb2d17765"))
