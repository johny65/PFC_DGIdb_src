import urllib.request
from urllib.error import URLError

url = "http://jcp.bmj.com/cgi/pmidlookup?view=long&pmid=25681512"

try:
    res = urllib.request.urlopen(url)
    if res.getcode() == 200:
        url = res.geturl()
except URLError as e:
    error = None

print(url)