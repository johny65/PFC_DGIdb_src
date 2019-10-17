import urllib.request
from urllib.error import URLError

url = 'http://www.thieme-connect.com/DOI/DOI?10.1055/s-0028-1094556'

try:
    res = urllib.request.urlopen(url)
    if res.getcode() == 200:
        url = res.geturl()
except URLError as e:
    error = None

print(url)