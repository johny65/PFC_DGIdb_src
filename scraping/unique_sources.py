import sys

def unique_sources(data_file):
    """data_file es un archivo de salida de base_pubmed.py, siendo cada línea 'pmid url'."""
    sources = set()
    with open(data_file) as f:
        for l in f:
            base_url = l.split(" ")[1]
            try:
                pos = base_url.index("://")
            except ValueError:
                continue
            pos += 3
            base_url = base_url[pos:base_url.index("/", pos)]
            sources.add(base_url)
        print("Fuentes distintas:", len(sources))
        print(sorted(list(sources)))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: {} entrada".format(sys.argv[0]))
        exit()
    unique_sources(sys.argv[1])
