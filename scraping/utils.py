import urllib.request
from urllib.error import HTTPError

def fetch_url(url):
    """Descarga y devuelve el contenido de una URL."""
    try:
        with urllib.request.urlopen(url) as f:
            data = f.read()
            return data.decode("utf8")
    except HTTPError:
        return ""