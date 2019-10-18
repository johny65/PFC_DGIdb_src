import urllib.request
from urllib.error import URLError

url = "https://diagnosticpathology.biomedcentral.com/articles/10.1186/s13000-016-0475-5"

try:
    res = urllib.request.urlopen(url)
    if res.getcode() == 200:
        url = res.geturl()
except URLError as e:
    error = None

print(url)