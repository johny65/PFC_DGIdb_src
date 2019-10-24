import sys

def lista_to_cadena(lista):
    '''
    Convierte una lista cuyos elementos son string a un único string
    '''
    lista[-1] = lista[-1].strip()
    cadena = ""
    contador = 0
    for elemento in lista:
        contador += 1
        if len(lista) == contador:
            cadena = cadena + elemento
        else:
            cadena = cadena + elemento + " "    
    return cadena

def datos_entrenamiento(ifg,abstracts_ruta,salida_tsv):
    '''
    Unifica los archivos ifg.csv y all_abstracts en uno solo con el formato de linea:
    pmid "\t" gen "\t" droga "\t" interaccion "\t" abstract "\n"
    '''
    for linea_ifg in ifg:
        datos = linea_ifg.split(",")
        pubmed = datos[0]
        gen = datos[1]
        droga = datos[2]
        interaccion = datos[3].strip()
        abstracts = open(abstracts_ruta,encoding="utf8")
        for linea_abs in abstracts:
            cadena = linea_abs.split()
            pmid = cadena[0]
            abstract_lista = cadena[1:]
            if pmid == pubmed:
                abstract_cadena = lista_to_cadena(abstract_lista)
                if abstract_cadena == "N/A":
                    break
                salida_tsv.write("{}\t{}\t{}\t{}\t{}\n".format(pubmed,gen,droga,interaccion,abstract_cadena))
                break
        abstracts.close()

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Forma de uso: {} entrada salida".format(sys.argv[0]))
        exit()

    ifg = open("D:\Descargas\Python\PFC_DGIdb_src\ifg.csv",encoding="utf8")
    abstract_ruta = "D:\Descargas\Python\PFC_DGIdb_src\Abstracts"
    salida_tsv = open("D:\Descargas\Python\PFC_DGIdb_src\datos_entrenamiento.tsv",'w',encoding="utf8")

    datos_entrenamiento(ifg,abstract_ruta,salida_tsv)

    ifg.close()