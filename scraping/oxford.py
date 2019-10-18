"""
Web scraping para Oxford Academic (academic.oup.com).
"""
from html.parser import HTMLParser
import urllib.request
import urllib.parse
import utils

class OxfordParser(HTMLParser):
    the_url = None
    a_attrs = None

    def __init__(self, url):
        HTMLParser.__init__(self)
        self.url = url
        headers = {"User-agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:69.0) Gecko/20100101 Firefox/69.0"}
        req = urllib.request.Request(url, headers=headers)
        res = urllib.request.urlopen(req)
        data = res.read().decode("utf8")
        self.feed(data)

    def handle_starttag(self, tag, attrs):
        if tag == "span" and ("class", "pdf-link-text") in attrs:
            for attr in self.a_attrs:
                if attr[0] == "href":
                    self.the_url = utils.get_server(self.url) + urllib.parse.unquote(attr[1])
        if tag == "a":
            self.a_attrs = attrs

def scrap(url):
    parser = OxfordParser(url)
    return parser.the_url

if __name__ == "__main__":
    print(scrap("https://academic.oup.com/jnci/article-lookup/doi/10.1093/jnci/dju121"))
