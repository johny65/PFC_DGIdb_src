"""
Script principal para el web scraping.
"""
import pathlib
import subprocess
import multiprocessing as mp
import sys
import math
import parallel
# scrapers:
# (cada scraper contiene un método 'scrap(url)' que dada la URL de la fuente
# devuelve la URL del PDF, o None si no pudo obtenerla)
import jbc
import cancerres
import jnccn
import nature
import springer
import oxford


def download_async(url, pmid):
    """Dada la URL de un PDF, lo descarga en files/pmid.pdf (usando wget en 
    forma asíncrona)."""
    out_file = "files/{}.pdf".format(pmid)
    subprocess.Popen(["wget", "-c", "-O", out_file, url])


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
    Dada una URL determina la fuente y llama al scraper correspondiente para que
    devuelve la URL para descargar el PDF.
    """
    if "jbc.org" in url:
        scraper = jbc
    elif "cancerres" in url:
        scraper = cancerres
    elif "jnccn.org" in url:
        scraper = jnccn
    elif "nature.com" in url:
        scraper = nature
    elif "link.springer.com" in url:
        scraper = springer
    elif "academic.oup.com" in url:
        scraper = oxford
    else:
        scraper = None
    
    return scraper.scrap(url) if scraper else None


def process_line(line):
    """Procesa una línea [pmid url]."""
    l = line.strip().split(" ")
    if len(l) > 1:
        pmid, url = l[0], " ".join(l[1:])
        if url != "No encontrado":
            print("Procesando", url)
            if pmid not in downloaded:
                pdf_url = process(url)
                if pdf_url:
                    download(pdf_url, pmid)
                    # download_async(pdf_url, pmid)
            else:
                print(pmid, "ya descargado")


if __name__ == "__main__":
    in_parallel = False
    if len(sys.argv) == 3 and sys.argv[2] == "-p":
        # ejecutar en paralelo
        in_parallel = True
    if len(sys.argv) < 2:
        print("Uso: {} entrada [-p]".format(sys.argv[0]))
        exit()
    downloaded = load_downloaded()

    with open(sys.argv[1]) as f:
        if in_parallel:
            parallel.parallel_map(process_line, f.readlines())
        else:
            for line in f:
                process_line(line)
