import sys
import os
import subprocess
    
def pdftotext(ruta_pdf, ruta_txt, for_windows):
    """
    ruta_pdf es la ruta de un directorio con los PDF.
    ruta_txt es la ruta de un directorio de salida donde guardar los txt.
    for_windows indica si se ejecuta en Windows.
    """
    if for_windows:
        xpdf_ruta = "E:/Descargas/Python/PFC_DGIdb_src/scraping/files/txt/pdftotext64.exe"
    else:
        xpdf_ruta = "pdftotext"
    
    for archivo in sorted(os.listdir(ruta_pdf)):
        if archivo.endswith(".pdf"):
            entrada_ruta = os.path.join(ruta_pdf,archivo)
            salida_nombre = archivo.replace(".pdf",".txt")
            salida_ruta = os.path.join(ruta_txt,salida_nombre)
            lista = [xpdf_ruta,"-enc","UTF-8","-nopgbrk","-nodiag",entrada_ruta,salida_ruta]
            # subprocess.Popen(lista)
            res = subprocess.run(lista)
            if res.returncode != 0:
                print("Error: " + archivo)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Modo de uso: {} entrada salida [-w]".format(sys.argv[0])) # argv[0] es el nombre de la función
        exit()

    for_windows = "-w" in sys.argv
    pdftotext(sys.argv[1], sys.argv[2], for_windows)
    