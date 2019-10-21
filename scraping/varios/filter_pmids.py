import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Forma de uso: {} entrada".format(sys.argv[0]))
        exit()

    pmids_usados = set()
    with open("pubs.csv") as f:
        for l in f:
            pmids_usados.add(l.strip())
    print("Cantidad de pmids usados:", len(pmids_usados))

    with open("dois_filtrados", "w") as out:
        with open(sys.argv[1]) as f:
            for l in f:
                pmid = l.split(",")[0]
                if pmid in pmids_usados:
                    out.write(l)