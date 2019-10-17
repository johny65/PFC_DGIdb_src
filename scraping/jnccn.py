"""
Web scraping para Journal of the National Comprehensive Cancer Network (jnccn.org).
"""
from html.parser import HTMLParser
import utils

class JnccnParser(HTMLParser):
    the_url = None
    a_attrs = None

    def __init__(self, url):
        HTMLParser.__init__(self)
        self.url = url
        data = utils.fetch_url_wget(url)
        self.feed(data)

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.a_attrs = attrs

    def handle_data(self, data):
        if data == "Download PDF to Print":
            for attr in self.a_attrs:
                if attr[0] == "href":
                    self.the_url = utils.get_base_server(self.url) + urllib.parse.unquote(attr[1])

def scrap(url):
    parser = JnccnParser(url)
    return parser.the_url

if __name__ == "__main__":
    print(scrap("https://jnccn.org/doi/10.6004/jnccn.2017.0139"))
    