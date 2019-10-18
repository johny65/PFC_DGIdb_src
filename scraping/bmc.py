"""
Web scraping para Bio Medical Central (www.biomedcentral.com).
"""
import urllib.request
from urllib.error import URLError

def scrap(url):
    try:
        res = urllib.request.urlopen(url)
        if res.getcode() == 200:
            # redirige a la misma url
            # si se reemplaza /articles por /track/pdf se obtiene el pdf
            # El archivo es descargado sin la extensión .pdf
            return res.geturl().replace("/articles", "/track/pdf")
    except URLError as e:
        return None

if __name__ == "__main__":
    url = "https://bmccancer.biomedcentral.com/articles/10.1186/s12885-016-2463-2"
    print(scrap(url))
    url = "https://arthritis-research.biomedcentral.com/articles/10.1186/ar398"
    print(scrap(url))
    url = "https://bmcpharma.biomedcentral.com/articles/10.1186/1471-2210-3-1"
    print(scrap(url))