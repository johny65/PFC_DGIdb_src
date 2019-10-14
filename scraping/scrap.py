"""
Script principal para el web scraping.
"""
import pathlib
import subprocess
import sys
# scrapers:
import jbc


def download(url, pmid):
    """Dada la URL de un PDF, lo descarga en files/pmid.pdf (usando wget)."""

    out_file = "files/{}.pdf".format(pmid)
    try:
        subprocess.run(["wget", "-c", "-O", out_file, url], check=True)
        print("Guardado", out_file)
    except subprocess.CalledProcessError as ex:
        print("Error descargando:", ex)


def load_downloaded():
    """Crea un caché con los pmids ya descargados para no volver a procesarlos."""
    cache = set()
    for p in pathlib.Path("files").iterdir():
        if p.is_file():
            cache.add(p.stem)
    return cache


def process(url):
    """
    Dada una URL determina la fuente y devuelve la URL para descargar el PDF según el
    scraper correspondiente.
    """
    if "jbc.org" in url:
        return jbc.jbc(url)
    return None


def process_line(line):
    """Procesa una línea [pmid url]."""
    l = line.strip().split(" ")
    if len(l) > 1:
        pmid, url = l[0], " ".join(l[1:])
        if url != "No encontrado":
            if pmid not in downloaded:
                pdf_url = process(url)
                if pdf_url:
                    download(pdf_url, pmid)
            else:
                print(pmid, "ya descargado")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: {} entrada")
        exit()
    downloaded = load_downloaded()
    with open(sys.argv[1]) as f:
        for l in f:
            process_line(l)
    