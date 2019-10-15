"""
Web scraping para Cancer Research (cancerres.aacrjournals.org) y Clinical Cancer Research
(clincancerres.aacrjournals.org).
"""
import urllib.request
from urllib.error import URLError

def scrap(url):
    # redirige a otra url que termina en .long
    # de la forma https://clincancerres.aacrjournals.org/content/21/14/3140.long
    # el pdf se obtiene cambiando 'content/' por 'content/clincanres/' y
    # '.long' por '.full-text.pdf'.
    # Para "cancerres" es lo mismo pero cambia a 'content/canres/'.
    if "clincancerres" in url:
        content = "clincanres"
    elif "cancerres" in url:
        content = "canres"
    try:
        res = urllib.request.urlopen(url)
        if res.getcode() == 200:
            return res.geturl().replace("content/", f"content/{content}/").replace(".long", ".full-text.pdf")
    except URLError as e:
        return None


if __name__ == "__main__":
    url = "http://clincancerres.aacrjournals.org/cgi/pmidlookup?view=long&pmid=26475333"
    print(scrap(url))
    url = "http://cancerres.aacrjournals.org/cgi/pmidlookup?view=long&pmid=27020857"
    print(scrap(url))
