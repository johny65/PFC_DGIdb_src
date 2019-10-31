import sys

def remove_cookie_absent():
    pmids = open("pmids")
    clean = open("pmids_clean")
    out = open("outs", "w")
    for l in clean:
        lorig = pmids.readline()
        if "cookieAbsent" in l:
            l = lorig
        out.write(l)
    out.close()
    pmids.close()
    clean.close()


def sin_abstract(infile):
    pmids = []
    with open(infile) as f:
        for l in f:
            if "N/A" == l.split()[1]:
                print(l)
                pmids.append(l.split()[0])
    print("Total N/A:", len(pmids))
    print("UPDATE publicaciones SET abstract = false WHERE pmid IN ({});".format(", ".join(pmids)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Falta archivo.")
        exit()
    sin_abstract(sys.argv[1])
