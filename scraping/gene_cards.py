"""
Web scraping de alias de genes de GeneCards.
"""
import sys
import utils
import parallel
import logging
from bs4 import BeautifulSoup

BASE_GENECARDS_URL = "https://www.genecards.org/cgi-bin/carddisp.pl?gene="


class GeneCardsParser():
    alias = []
    def fetch(self, gene):
        """Descarga el contenido de la página. NECESITA COOKIES PARA QUE ANDE."""
        url = BASE_GENECARDS_URL + gene
        logging.debug("Fetching URL " + url)
        headers = {
            "User-agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:69.0) Gecko/20100101 Firefox/69.0",
            "Cookie": "ASP.NET_SessionId=hwjs2brcx2cbpjod4gnxijlv; rvcn=H1d-sbuapwNh7VvK5OojnZx_GNvK8NV3Y_igQlT3X1auxx-CGR_50Kv2Gtv0Wo_OSexSGqUeuYFqw_sxZCt8GPagEGA1; ARRAffinity=166bde02ef81ff7e7ac9e9a57f0ef302100f353e9212ba930c859133d8b6d672; visid_incap_146342=ng+3dLHhQg+n22cnTIdnpyPQzV0AAAAAQUIPAAAAAADYkswfNl1m+lEO6s1+k/62; nlbi_146342=67fBdWPizFQhuWCUmewSQgAAAACX5WRNOa574GkShdKAhsHo; incap_ses_789_146342=QK2DWJBMgCysSvfvbRjzCiTQzV0AAAAA6eTm5xFAT0bApBqNQRJH0w==; _ga=GA1.2.752885262.1573769256; _gid=GA1.2.1465475919.1573769256; __gads=ID=bb532cbe1d9196bc:T=1573769276:S=ALNI_MZlxaQcjBdoHS5r7fq1pdgh3l_5cg; EU_COOKIE_LAW_CONSENT=true"
        }
        data = utils.fetch_url(url, headers)
        return data

    def __init__(self, gene, data=None):
        if not data:
            data = self.fetch(gene)
        soup = BeautifulSoup(data, "html.parser")
        try:
            section = soup.find(id="aliases_descriptions").find(class_="gc-subsection")
            for ul in section("ul"):
                for li in ul("li"):
                    logging.debug(li)
                    alias = li.contents[0].strip()
                    if not alias: #está en un span
                        alias = li.contents[1].text.strip()
                    self.alias.append(alias)
            self.alias.sort()
        except Exception as ex:
            logging.error(ex)
            print("No se encontraron alias para", gene)


if __name__ == "__main__":
    # descomentar para activar logs para debug:
    # logging.getLogger().setLevel(logging.DEBUG)
    parser = GeneCardsParser("ABAT")
    print("Cantidad de alias:", len(parser.alias))
    print("Alias:", parser.alias)
