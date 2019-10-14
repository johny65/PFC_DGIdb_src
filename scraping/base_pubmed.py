"""
URL scraping de las publicaciones de PubMed. Dado un archivo con los diferentes PMID por línea,
obtiene la URL de las fuentes y las guarda al lado de cada PMID.
"""

import urllib.request
from urllib.error import HTTPError
import sys

def fetch_url(url):
    """Descarga y devuelve el contenido de una URL."""
    try:
        with urllib.request.urlopen(url) as f:
            data = f.read()
            return data.decode("utf8")
    except HTTPError:
        return "404"

def get_src_url(pmid):
    """Dado un PMID, devuelve la URL de la fuente."""
    BASE_PUBMED_URL = "https://www.ncbi.nlm.nih.gov/pubmed/"
    data = fetch_url(BASE_PUBMED_URL + pmid)
    return extract_src_url(data)

def extract_src_url(data):
    """Extrae la URL de la fuente a partir del contenido de una página de PubMed."""
    for l in data.split("\n"):
        try:
            found = l.index("Full Text Sources")
            href = l.index("href=", found)
            url = l[href+6:l.index('"', href+7)].replace("&amp;", "&")
            return url
        except:
            pass
    return "No encontrado"

def test():
    """Función para verificar funcionamiento."""
    data = open("example.html").read()
    assert "https://linkinghub.elsevier.com/retrieve/pii/S0002-9149(99)00490-7" == extract_src_url(data)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "-t":
        test()
        exit()
    if len(sys.argv) != 3:
        print("Uso: {} entrada salida".format(sys.argv[0]))
        exit()
    with open(sys.argv[2], "w") as out:
        with open(sys.argv[1]) as f:
            for l in f:
                if len(l.split()) > 1:
                    out.write(l)
                else:
                    url = get_src_url(l)
                    print(l.strip(), url)
                    out.write(l.strip() + " " + url + "\n")
