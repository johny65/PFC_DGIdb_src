"""
Web scraping para ASPET: The Journal of Pharmacology and Experimental Therapeutics
(jpet.aspetjournals.org), Molecular Pharmacology (molpharm.aspetjournals.org),
Drug Metabolism and Disposition (dmd.aspetjournals.org) y Pharmacological Reviews
(pharmrev.aspetjournals.org).
"""
import utils

def scrap(url):
    # redirige a otra url que termina en .long
    suffix = ".full-text.pdf"
    if "jpet" in url:
        content = "jpet"
    elif "molpharm" in url:
        content = "molpharm"
    elif "dmd" in url:
        content = "dmd"
    elif "pharmrev" in url:
        content = "pharmrev"
        suffix = ".full.pdf"
    res = utils.redirect(url)
    return res.replace("content/", f"content/{content}/").replace(".long", suffix) or None

if __name__ == "__main__":
    print(scrap("http://jpet.aspetjournals.org/cgi/pmidlookup?view=long&pmid=1309873"))
    print(scrap("http://molpharm.aspetjournals.org/cgi/pmidlookup?view=long&pmid=2704370"))
    print(scrap("http://dmd.aspetjournals.org/cgi/pmidlookup?view=long&pmid=2903022"))
    print(scrap("http://pharmrev.aspetjournals.org/cgi/pmidlookup?view=long&pmid=16507884"))
