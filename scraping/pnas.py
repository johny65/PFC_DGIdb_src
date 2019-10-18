"""
Web scraping para Proceedings of the National Academy of Sciences of the United States of
America (www.pnas.org).
"""
import utils

def scrap(url):
    # redirige a otra url que termina en .long
    res = utils.redirect(url)
    return res.replace("content/", "content/pnas/").replace(".long", ".full.pdf") if res else None

if __name__ == "__main__":
    print(scrap("http://www.pnas.org/cgi/pmidlookup?view=long&pmid=10339548"))
