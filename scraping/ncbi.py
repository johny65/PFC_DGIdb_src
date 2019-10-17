"""
Web scraping para National Center for Biotechnology Information (www.ncbi.nlm.nih.gov).
"""
import urllib.request
from urllib.error import URLError

def scrap(url):
    try:
        res = urllib.request.urlopen(url)
        if res.getcode() == 200:
            # redirige a otra url que termina en .long
            # si se reemplaza .long por .full.pdf se obtiene el pdf
            return res.geturl().replace(".long", ".full.pdf")
    except URLError as e:
        return None
