"""
Super scraper.
"""
import urllib.request
import urllib.parse
from urllib.error import URLError
import utils

def scrap(pmid):
    data = urllib.parse.urlencode({"request": pmid})
    url = "https://sci-hub.se"
    req = urllib.request.Request(url, method="POST", data=bytes(data, "utf8"))
    res = urllib.request.urlopen(req)

    contents = res.read().decode("utf8")
    try:
        pos = contents.index("location.href")
        return contents[pos+15:contents.index("'", pos+16)]
    except:
        return None

    # headers = {"User-agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:69.0) Gecko/20100101 Firefox/69.0",
    # "Cookie": "__cfduid=de80d2e71b930c74f22545c4b9d14b28e1573158649; SERVER=WZ6myaEXBLEFIVkvFByAkw==; MAID=ckXfHIyxCnMSnPlwNXBugQ==; MACHINE_LAST_SEEN=2019-11-07T12%3A30%3A51.254-08%3A00; JSESSIONID=aaavUvR-cufYK5cR6qe5w; AMCV_1B6E34B85282A0AC0A490D44%40AdobeOrg=-1303530583%7CMCIDTS%7C18208%7CMCMID%7C32827639548205788921063899123516055662%7CMCAAMLH-1573763456%7C4%7CMCAAMB-1573763456%7C6G1ynYcLPuiQxYZrsz_pkqfLG9yMXBpb2zX5dvJdYQJzPXImdj0y%7CMCOPTOUT-1573165856s%7CNONE%7CMCAID%7CNONE%7CMCSYNCSOP%7C411-18215%7CvVersion%7C3.3.0; _ga=GA1.2.1020522758.1573158655; _gid=GA1.2.1821080210.1573158655; _fbp=fb.1.1573158655486.2082769408; AMCVS_1B6E34B85282A0AC0A490D44%40AdobeOrg=1; __gads=ID=3c79d20520bce333:T=1573158655:S=ALNI_MbcrwslNHBl_q6qRMJiIYcA4UKZcg; s_cc=true; s_sq=%5B%5BB%5D%5D; _sdsat_MCID=32827639548205788921063899123516055662; randomizeUser=0.21920599212661052; __atuvc=5%7C45; __atuvs=5dc47f4da5809ebc004"}
    # req = urllib.request.Request(url, headers=headers)
    # res = urllib.request.urlopen(req)
    # return res.geturl().replace("/abs/", "/pdfdirect/")

def scrap2(doi):
    base_url = "http://libgen.lc/scimag/ads.php?doi="
    html = utils.fetch_url(base_url + doi)
    for fila in html.split("\n"):
        try:
            clave = fila.index("http://booksdl.org/scimag/get.php?doi=" + doi)
            url = fila[clave:fila.index('"',clave)]
            return url
        except:
            pass

if __name__ == "__main__":
    print(scrap("25864487"))
    print(scrap2("10.1016/0003-9861(87)90443-7"))