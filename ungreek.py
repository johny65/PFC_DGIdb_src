import sys
import pathlib

greek_letters = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega"
}

def ungreek(in_file, out_dir):
    with open(in_file, encoding="utf8") as f:
        contents = _ungreek(f.read())
    if contents:
        out_file = out_dir / in_file.name
        with open(out_file, "w", encoding="utf8") as f:
            f.write(contents)


def _ungreek(contents):
    """Reemplaza las letras griegas por sus nombres (convierte todo a minúsculas)."""
    contents = contents.lower()
    for letter, name in greek_letters.items():
        contents = contents.replace(letter, name)
    return contents


def _clean_html(contents):
    """Elimina etiquetas HTML del contenido."""
    for letter, name in greek_letters.items():
        contents = contents.replace("&" + name + ";", name)
    for tag in ["<sub>", "</sub>", "<sup>", "</sup>", "<i>", "</i>", "<a>", "</a>"]:
        contents = contents.replace(tag, "")
    contents = contents.replace("&gt;", ">")
    contents = contents.replace("&lt;", "<")
    contents = contents.replace("&amp;", "&")
    contents = contents.replace("&mgr;", "mu")
    contents = contents.replace("&uuml;", "ü")
    return contents


if __name__ == "__main__":
    if len(sys.argv) < 3:
        exit("Modo de uso: {} dir_entrada dir_salida".format(sys.argv[0]))
    
    out_dir = pathlib.Path(sys.argv[2])
    if not out_dir.exists():
        out_dir.mkdir()
    for p in pathlib.Path(sys.argv[1]).iterdir():
        if p.is_file():
            ungreek(p, out_dir)

if __name__ == "__main__2":
    if len(sys.argv) < 2:
        exit("Modo de uso: {} in_file".format(sys.argv[0]))
    with open(sys.argv[1]) as f:
        c = f.read()
        c = _ungreek(c)
        c = _clean_html(c)
    print(c)
