import utils

with open("pmids") as f:
    for l in f:
        pmid, url = l.split()[0], l.split()[1]
        server = utils.get_server(url) if url != "No" else "-"
        print("{},{}".format(pmid, server))
