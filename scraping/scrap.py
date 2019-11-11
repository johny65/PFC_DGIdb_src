"""
Script principal para el web scraping.
"""
import pathlib
import subprocess
import multiprocessing as mp
import sys
import math
import parallel
import logging
from termcolor import colored
# scrapers:
# (cada scraper contiene un método 'scrap(url)' que dada la URL de la fuente
# devuelve la URL del PDF, o None si no pudo obtenerla)
import jbc
import cancerres
import jnccn
import nature
import springer
import oxford
import bmc
import aspet
import pnas
import physiology
import wiley
import super_scrap

# scrapers activos, sólo se usarán los scrapers de esta lista (si está vacía se usan todos):
ACTIVE_SCRAPERS = []
# los scrapers de la siguiente lista siempre se excluirán:
EXCLUDE_SCRAPERS = []

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
        print(colored("Guardado", "green"), out_file)
    except subprocess.CalledProcessError as ex:
        print(colored("Error descargando.", "red"))
        logging.log(logging.ERROR, "Error descargando:", ex)


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
    elif "biomedcentral.com" in url:
        scraper = bmc
    elif "aspetjournals.org" in url:
        scraper = aspet
    elif "pnas.org" in url:
        scraper = pnas
    elif "physiology.org" in url:
        scraper = physiology
    elif "wiley" in url:
        scraper = wiley
    else:
        scraper = None
    
    # activo?
    active = (not ACTIVE_SCRAPERS or scraper in ACTIVE_SCRAPERS) \
        and scraper not in EXCLUDE_SCRAPERS
    return scraper.scrap(url) if scraper and active else None


def process_line(line):
    """Procesa una línea [pmid url]."""
    downloaded = load_downloaded()
    use_super = True
    l = line.strip().split(" ")
    if len(l) > 1:
        pmid, url = l[0], " ".join(l[1:])
        if url != "No encontrado":
            if pmid not in downloaded:
                if use_super:
                    # pdf_url = super_scrap.scrap(pmid)
                    pdf_url = super_scrap.scrap2(url)
                else:
                    pdf_url = process(url)

                if pdf_url:
                    print("Descargando", pdf_url)
                    download(pdf_url, pmid)
                    # download_async(pdf_url, pmid)
            else:
                print(pmid, colored("ya descargado", "yellow"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: {} entrada [-p] [--super]".format(sys.argv[0]))
        exit()
    downloaded = load_downloaded()

    in_parallel = "-p" in sys.argv
    use_super = "--super" in sys.argv

    with open(sys.argv[1]) as f:
        if in_parallel:
            parallel.parallel_map(process_line, f.readlines())
        else:
            for line in f:
                process_line(line)
