"""
Web scraping de abstracts.
"""
from html.parser import HTMLParser
import utils

BASE_PUBMED_URL = "https://www.ncbi.nlm.nih.gov/pubmed/"

class AbstractParser(HTMLParser):
    the_url = None
    in_div = False
    in_p = False
    abstract = None

    def __init__(self, pmid):
        HTMLParser.__init__(self)
        url = BASE_PUBMED_URL + pmid
        data = utils.fetch_url_wget_windows("D:\Descargas\wget.exe", url)
        self.feed(data)

    def handle_starttag(self, tag, attrs):
        self.in_div = (tag == "div" and ("class", "abstr") in attrs)
        self.in_p = (tag == "p" and self.in_div)
        
    def handle_data(self, data):
        if self.in_p:
            self.abstract = data


def scrap(pmid):
    parser = AbstractParser(pmid)
    return parser.abstract


if __name__ == "__main__":
    print(scrap("23917377"))
