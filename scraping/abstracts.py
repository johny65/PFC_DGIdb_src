"""
Web scraping de abstracts.
"""
from html.parser import HTMLParser
import sys
import utils
import parallel
try:
    from bs4 import BeautifulSoup
except:
    print("No se pudo importar BeautifulSoup")

BASE_PUBMED_URL = "https://www.ncbi.nlm.nih.gov/pubmed/"

class AbstractParser(HTMLParser):
    in_div = False
    in_p = False
    abstract = None

    def __init__(self, pmid):
        HTMLParser.__init__(self)
        url = BASE_PUBMED_URL + pmid
        data = utils.fetch_url_wget(url)
        self.feed(data)

    def handle_starttag(self, tag, attrs):
        if tag == "div" and ("class", "abstr") in attrs:
            self.in_div = True
        self.in_p = (tag == "p" and self.in_div)
        
    def handle_data(self, data):
        if self.in_p and not self.abstract:
            self.abstract = data.replace("\n", " ")


class BSAbstractParser():
    def __init__(self, pmid):
        url = BASE_PUBMED_URL + pmid
        data = utils.fetch_url_wget(url)
        soup = BeautifulSoup(data, 'html.parser')
        d = soup.find(class_="abstr")
        self.abstract = d.get_text(" ")


def load_downloaded(outfile):
    """Crea un caché con los pmids de los abstracts ya descargados para no volver a procesarlos."""
    cache = set()
    try:
        with open(outfile) as f:
            for l in f:
                # pmid abstract
                cache.add(l.split()[0])
    except OSError:
        pass
    print("Ya existen descargados", len(cache), "abstracts.")
    return cache


def process_line(line):
    pmid = line.split()[0]
    if pmid in downloaded:
        # print(pmid, "ya descargado")
        pass
    else:
        print("Procesando", pmid)
        # parser = AbstractParser(pmid)
        parser = BSAbstractParser(pmid)
        return pmid + " " + (parser.abstract or "N/A")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: {} entrada salida descargados".format(sys.argv[0]))
        exit()
    downloaded = load_downloaded(sys.argv[3])
    with open(sys.argv[1]) as f:
        parallel.parallel_map_to_file(process_line, f.readlines(), sys.argv[2])
