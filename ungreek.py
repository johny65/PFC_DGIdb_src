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
        contents = f.read()
        for letter, name in greek_letters.items():
            contents = contents.replace(letter, name)
    
    out_file = out_dir / in_file.name
    with open(out_file, "w", encoding="utf8") as f:
        f.write(contents)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        exit("Modo de uso: {} dir_entrada dir_salida".format(sys.argv[0]))
    
    out_dir = pathlib.Path(sys.argv[2])
    if not out_dir.exists():
        out_dir.mkdir()
    for p in pathlib.Path(sys.argv[1]).iterdir():
        if p.is_file():
            ungreek(p, out_dir)
    