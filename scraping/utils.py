from html.parser import HTMLParser
from urllib.error import HTTPError
import urllib.request
import urllib.parse

class ElsevierParser(HTMLParser):
    """Parseador para obtener la URL de redirección de un enlace de Elsevier."""
    the_url = None
    def handle_starttag(self, tag, attrs):
        if tag == "input" and ("type", "hidden") in attrs and ("name", "redirectURL") in attrs:
            for attr in attrs:
                if attr[0] == "value":
                    self.the_url = urllib.parse.unquote(attr[1])

def fetch_url(url):
    """Descarga y devuelve el contenido de una URL."""
    try:
        with urllib.request.urlopen(url) as f:
            data = f.read()
            return data.decode("utf8")
    except HTTPError:
        return ""

def redirect(url):
    """Navega la URL y devuelve la URL a la que redirigió."""
    try:
        res = urllib.request.urlopen(url)
        return res.geturl()
    except:
        return None

def elsevier(url):
    """Obtiene el verdadero enlace de la publicación desde un enlace de Elsevier."""
    data = fetch_url(url)
    parser = ElsevierParser()
    parser.feed(data)
    return parser.the_url

def get_base_server(url):
    """Devuelve el servidor de un enlace. `None` si la url no tiene el protocolo."""
    try:
        pos = url.index("://") + 3
        return url[pos:url.index("/", pos)]
    except:
        return None
