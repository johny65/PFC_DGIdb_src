import sys
from utils import fetch_url
import csv

def get_doi(pmid):
    '''
    Dado un pmid extrae su DOI de Pubmed si este existe.
    '''
    pubmed_url = "https://www.ncbi.nlm.nih.gov/pubmed/"
    html = fetch_url(pubmed_url + pmid)
    for fila in html.split("\n"):
        try:
            doi_key = fila.index("DOI:")
            href = fila.index('href="',doi_key)
            doi = fila[href+16:fila.index('"',href+16)]
            return doi
        except:
            pass

def get_titulo(pmid):
    '''
    Dado un pmid extrae su título de Pubmed si este existe.
    '''
    pubmed_url = "https://www.ncbi.nlm.nih.gov/pubmed/"
    html = fetch_url(pubmed_url + pmid)
    for fila in html.split("\n"):
        try:
            clave = fila.index("<title>")
            titulo = fila[clave+7:fila.index('- PubMed - NCBI',clave+7)]
            return titulo
        except:
            pass

def get_abstract(pmid):
    '''
    Dado un pmid extrae su abstract de Pubmed si este existe.
    '''
    pubmed_url = "https://www.ncbi.nlm.nih.gov/pubmed/"
    html = fetch_url(pubmed_url + pmid)
    for fila in html.split("\n"):
        try:
            clave = fila.index('"abstr"')
            p = fila.index("<p>",clave)
            abstract = fila[p+3:fila.index('</p>',p+3)]
            return abstract
        except:
            pass

def get_keywords(pmid):
    '''
    Dado un pmid extrae su palabras clave de Pubmed si estas existen.
    '''
    pubmed_url = "https://www.ncbi.nlm.nih.gov/pubmed/"
    html = fetch_url(pubmed_url + pmid)
    for fila in html.split("\n"):
        try:
            clave = fila.index("KEYWORDS:")
            p = fila.index("<p>",clave)
            abstract = fila[p+3:fila.index('</p>',p+3)]
            return abstract
        except:
            pass

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Modo de uso: {} entrada salida".format(sys.argv[0])) # argv[0] es el nombre de la función
        exit()

    with open(sys.argv[2],"w",encoding="utf8") as pmids_dois:
        escritor_csv = csv.writer(pmids_dois,delimiter=',',lineterminator="\n")
        with open(sys.argv[1],encoding="utf8") as pmids:
            lector_csv = csv.reader(pmids,delimiter=',',quoting=csv.QUOTE_ALL)
            c = 1
            for fila in lector_csv:
                print(c)
                c += 1 
                pmid = fila[0]
                # doi = get_doi(pmid)
                titulo = get_titulo(pmid)
                if titulo is None:
                    titulo = ""
                abstract = get_abstract(pmid)
                if abstract is None:
                    abstract = ""
                keywords = get_keywords(pmid)
                if keywords is None:
                    keywords = ""
                resultado = titulo + abstract + keywords
                # lista = [pmid, doi]
                lista = [pmid, resultado]
                print(resultado)
                # print(lista)
                escritor_csv.writerow(lista)