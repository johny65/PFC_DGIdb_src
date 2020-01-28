import pathlib
import shutil
import subprocess
import sys
from bs4 import BeautifulSoup

def ocr(filename):
    # filename es la ruta a un archivo PDF (objeto pathlib);
    # primero se debe transformar a imágenes
    p = filename
    subprocess.run(["pdfimages", p.name, p.stem])
    imgs = [f for f in cwd.glob(p.stem + "*.p*m")]
    for img in imgs:
        subprocess.run(["tesseract", img.name, img.stem])
    txts = [f.name for f in cwd.glob(p.stem + "*.txt")]
    if txts:
        res = subprocess.run(["cat"] + txts, capture_output=True, text=True)
        with open("ocr/{}.txt".format(p.stem), "w") as out:
            out.write(res.stdout)


if __name__ == "__main__":
    cwd = pathlib.Path()
    txt_dir = cwd / "txt_ungreek"
    ocr_dir = cwd / "ocr"

    # de los pdfs que no tengan txt hacer el ocr
    for pdf in cwd.glob("*.pdf"):
        txt = txt_dir / (pdf.stem + ".txt")
        ocred = ocr_dir / (pdf.stem + ".txt")
        if not txt.exists() and not ocred.exists():
            res = subprocess.run(["file", pdf], capture_output=True, text=True)
            # si son html sacarlos nomás
            if "HTML document" in res.stdout:
                # shutil.copy(pdf, ocr_dir / (pdf.stem + ".htm"))
                pass
            else:
                print("Falta txt de", pdf.name)
                # ocr(pdf)
                pass

if __name__ == "__main__2":
    filename = sys.argv[1]
    with open(filename) as f:
        data = f.read()
    soup = BeautifulSoup(data, 'html.parser')
    print(soup.get_text(" "))
