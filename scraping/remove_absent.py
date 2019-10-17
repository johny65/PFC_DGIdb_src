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