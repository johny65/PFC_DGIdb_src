import urllib.request
from urllib.error import URLError

url = "https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/393437/"

try:
    res = urllib.request.urlopen(url)
    if res.getcode() == 200:
        url = res.geturl()
except URLError as e:
    error = None

print(url)