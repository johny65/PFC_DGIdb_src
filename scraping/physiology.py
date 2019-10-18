"""
Web scraping para American Physiological Society (www.physiology.org).
"""
from html.parser import HTMLParser
import urllib.parse
import utils

class PhysiologyParser(HTMLParser):
    the_url = None
    a_attrs = None
    in_a = False

    def __init__(self, url):
        HTMLParser.__init__(self)
        data = utils.fetch_url_wget(url)
        self.feed(data)

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.in_a = True
            self.a_attrs = attrs
        else:
            self.in_a = False

    def handle_data(self, data):
        if self.in_a and data == "Download PDF":
            for attr in self.a_attrs:
                if attr[0] == "href":
                    self.the_url = urllib.parse.unquote(attr[1])

def scrap(url):
    parser = PhysiologyParser(url)
    return parser.the_url

if __name__ == "__main__":
    print(scrap("http://www.physiology.org/doi/full/10.1152/ajplung.00393.2003?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%3dpubmed"))
