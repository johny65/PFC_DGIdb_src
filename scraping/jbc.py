import urllib.request
from urllib.error import URLError

def fetch_url(url):
    try:
        res = urllib.request.urlopen(url)
        if res.getcode() == 200:
            # redirige a otra url que termina en .long
            # si se reemplaza .long por .full.pdf se obtiene el pdf
            res = urllib.request.urlopen(res.geturl().replace(".long", ".full.pdf"))
            return res.read()
    except URLError as e:
        print("Error", e)

def jbc(url, pmid):
    out_file = "files/{}.pdf".format(pmid)
    with open(out_file, "w+b") as out:
        out.write(fetch_url(url))
        print("jbc: Guardado", out_file)
