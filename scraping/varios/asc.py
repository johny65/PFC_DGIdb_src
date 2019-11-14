import sys
import csv
import utils

def pmid_src():
    """Guarda en un archivo el pmid y su fuente."""
    with open("pmids") as f:
        for l in f:
            pmid, url = l.split()[0], l.split()[1]
            server = utils.get_server(url) if url != "No" else "-"
            print("{},{}".format(pmid, server))


def getCsvReader(reading_file):
    return csv.reader(reading_file, delimiter=',', quoting=csv.QUOTE_ALL)


def generate_training(ifg_file, abstracts_file, out_file):
    """
    Genera un archivo donde cada fila es:
    [pmid, gen, droga, interacción, abstract]
    Ejemplo: 10722,ADRA1A,MEPHENTERMINE,agonist,el abstract
    """
    abstracts = {}
    with open(abstracts_file, encoding="utf8") as afile:
        for l in afile:
            ll = l.split()
            pmid = ll[0]
            a = " ".join(ll[1:])
            abstracts[pmid] = a.strip()

    out = open(out_file, "w", encoding="utf8")
    writer = csv.writer(out, delimiter='\t', lineterminator="\n")

    with open(ifg_file, encoding="utf8") as csvfile:
        reader = getCsvReader(csvfile)
        for row in reader:
            out_line = list(row) + [abstracts[row[0]]]
            writer.writerow(out_line)
    out.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("ifg_file, abstracts_file, out_file")
        exit()
    generate_training(sys.argv[1], sys.argv[2], sys.argv[3])
