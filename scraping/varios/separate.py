import pathlib
import sys

def separate(in_file, orig_dir, dest_dir):
    """"A partir de los pmids nombrados en 'in_file', los toma de 'orig_dir' y los mueve a 'dest_dir'."""
    files = set()
    with open(in_file, encoding="utf8") as f:
        for l in f:
            files.add(l.split()[0])
    
    dest = pathlib.Path(dest_dir)
    if not dest.exists():
        dest.mkdir()
    
    for p in pathlib.Path(orig_dir).iterdir():
        if p.stem in files:
            print("Moviendo", p.name)
            p.rename(dest / p.name)
    
if __name__ == "__main__":
    if len(sys.argv) != 4:
        exit("Modo de uso: {} entrada dir_origen dir_destino".format(sys.argv[0]))
    separate(sys.argv[1], sys.argv[2], sys.argv[3])
