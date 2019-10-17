import utils
import sys

def unique_sources(data_file):
    """data_file es un archivo de salida de base_pubmed.py, siendo cada línea 'pmid url'."""
    sources = {}
    with open(data_file) as f:
        for l in f:
            base_url = utils.get_server(l.split(" ")[1])
            if not base_url: continue
            if not base_url in sources:
                sources[base_url] = 0
            sources[base_url] += 1
        print("Fuentes distintas:", len(sources))
        print("Artículos en cada una:")
        for x in sorted(list(sources.items()), key=lambda x: x[1], reverse=True):
            print(x[0] + ":", x[1])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: {} entrada".format(sys.argv[0]))
        exit()
    unique_sources(sys.argv[1])
