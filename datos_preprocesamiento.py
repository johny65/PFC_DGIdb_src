import sys
import csv
import numpy as np
import re
import os

# Preprocesamiento de los datos (texto)
# from keras.preprocessing import text,sequence 

def cargar_pmids(pmids_etiquetas_completas_csv):
    '''
    Carga los pmids con las etiquetas completas en una lista.
    '''
    pmids_lista = list()
    with open(pmids_etiquetas_completas_csv,encoding="utf8") as pmids_csv:
        lector_csv = csv.reader(pmids_csv,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            pmids_lista.append(fila[0])
    return pmids_lista

def cargar_aliases_conjunto(aliases_ruta):
    '''
    Carga los aliases de gen o de droga dependiendo del archivo de entrada en un conjunto
    '''
    aliases_conjunto = set()
    with open(aliases_ruta,encoding="utf8") as aliases_csv:
        lector_csv = csv.reader(aliases_csv,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            for e in fila:
                aliases_conjunto.add(e.lower())
    return aliases_conjunto

def cargar_aliases_lista(aliases_ruta):
    '''
    Carga los aliases de gen o de droga dependiendo del archivo de entrada en una lista
    '''
    aliases_lista = list()
    with open(aliases_ruta,encoding="utf8") as aliases_csv:
        lector_csv = csv.reader(aliases_csv,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            aliases_lista.append(fila)
    return aliases_lista

def cargar_embeddings(embeddings_ruta):
    '''
    Carga los vectores de embeddings de GloVe
    '''
    embeddings_dict = dict()
    with open(embeddings_ruta,encoding="utf8") as embeddings:
        for fila in embeddings:
            datos = fila.split()
            palabra = datos[0]
            embedding = np.asarray(datos[1:],dtype='float32')
            embeddings_dict[palabra] = embedding
    return embeddings_dict

def cargar_ifg(ifg_csv_ruta):
    '''
    Carga las interacciones fármaco-gen en una lista.
    '''
    ifg_lista = list()
    with open(ifg_csv_ruta,encoding="utf8") as ifg_csv:
        lector_csv = csv.reader(ifg_csv,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            ifg_lista.append(fila)
    return ifg_lista

def cargar_repeticiones(repeticiones_ruta):
    '''
    Carga los genes/drogas repetidos
    '''
    repeticiones_lista = list()
    with open(repeticiones_ruta,encoding="utf8") as repeticiones:
        lector_csv = csv.reader(repeticiones,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            repeticiones_lista.append(fila[0])
    return repeticiones_lista

def aliases_repeticiones(aliases_conjunto,aliases_lista,salida):
    '''
    Guarda en un archivo los aliases de genes/drogas repetidos junto a la cantidad de repeticiones
    '''
    with open(salida,"w",encoding="utf8") as salida_csv:
        escritor_csv = csv.writer(salida_csv,delimiter=',',lineterminator="\n")
        for elemento in aliases_conjunto:
            lista = list()
            contador = 0
            for entidad in aliases_lista:
                if elemento in entidad:
                    contador += 1
            if contador > 1:
                lista.append(elemento)
                lista.append(contador)
                escritor_csv.writerow(lista)

def cargar_publicaciones(publicaciones_directorio,pmids_titulos_abstracts_keywords_ruta,pmids_lista):
    '''
    Carga las publicaciones que ya se encuentran en formato txt.
    Si la publicación no existe carga el titulo_abstract_keywords en su lugar.
    '''
    publicaciones_dict = dict()
    for pmid in pmids_lista:
        archivo_nombre = pmid + ".txt"
        if archivo_nombre in sorted(os.listdir(publicaciones_directorio)):
            archivo_ruta = os.path.join(publicaciones_directorio,archivo_nombre)
            with open(archivo_ruta,encoding="utf8") as publicacion:
                publicaciones_dict[pmid] = publicacion.read().lower()
        else:
            with open(pmids_titulos_abstracts_keywords_ruta,encoding="utf8") as abstracts:
                lector_csv = csv.reader(abstracts,delimiter=',',quoting=csv.QUOTE_ALL)
                for linea in lector_csv:
                    if linea[0] == pmid:
                        publicaciones_dict[pmid] = linea[1].lower()
    return publicaciones_dict

def ocurrencias(entidades,publicaciones,embeddings,repeticiones,salida):
    '''
    Muestra las apariciones de genes/aliases o drogas/aliases en las publicaciones que no tienen embedding
    '''
    with open(salida,"w",encoding="utf8") as salida_csv:
        escritor_csv = csv.writer(salida_csv,delimiter=',',lineterminator="\n")
        for elemento in publicaciones:
            print(elemento)
            lista = list()
            for entidad in entidades:
                if entidad in publicaciones[elemento] and entidad not in embeddings and entidad not in repeticiones:
                    lista.append(entidad)
            escritor_csv.writerow(lista)

def etiquetas_publicacion_gen(pmids_lista,ifg_lista,aliases_lista,salida):
    with open(salida,"w",encoding="utf8") as salida_csv:
        escritor_csv = csv.writer(salida_csv,delimiter=',',lineterminator="\n")
        lista = list()
        for pmid in pmids_lista:
            lista = [pmid]
            for etiqueta in ifg_lista:
                if pmid == etiqueta[0]:
                    for entidad in aliases_lista:
                        if etiqueta[1].lower() == entidad[0].lower():
                            lista = lista + entidad
            print(lista)
            escritor_csv.writerow(lista)

def etiquetas_publicacion_droga(pmids_lista,ifg_lista,aliases_lista,salida):
    with open(salida,"w",encoding="utf8") as salida_csv:
        escritor_csv = csv.writer(salida_csv,delimiter=',',lineterminator="\n")
        lista = list()
        for pmid in pmids_lista:
            lista = [pmid]
            for etiqueta in ifg_lista:
                if pmid == etiqueta[0]:
                    for entidad in aliases_lista:
                        if etiqueta[2].lower() == entidad[0].lower():
                            lista = lista + entidad
            print(lista)
            escritor_csv.writerow(lista)

if __name__ == "__main__":
    # if len(sys.argv) != 1:
    #     print("Forma de uso: {} entrada salida".format(sys.argv[0]))
    #     exit()
    
    pmids_etiquetas_completas_csv = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/pmids_etiquetas_completas.csv"
    pmids_lista = cargar_pmids(pmids_etiquetas_completas_csv)
    # print(pmids_lista)
    print("pmids cargados")

    aliases_gen_ruta = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/alias_gen.csv"
    aliases_gen_conjunto = cargar_aliases_conjunto(aliases_gen_ruta)
    # aliases_gen_lista = cargar_aliases_lista(aliases_gen_ruta)
    # print(aliases_gen)
    print("alias gen cargados")

    aliases_droga_ruta = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/alias_droga.csv"
    aliases_droga_conjunto = cargar_aliases_conjunto(aliases_droga_ruta)
    # aliases_droga_lista = cargar_aliases_lista(aliases_droga_ruta)
    # print(aliases_droga)
    print("alias droga cargados")

    repeticiones_genes_ruta = "E:/Descargas/Python/PFC_DGIdb_src/repeticiones_genes.csv"
    repeticiones_drogas_ruta = "E:/Descargas/Python/PFC_DGIdb_src/repeticiones_drogas.csv"
    repeticiones_genes_lista = cargar_repeticiones(repeticiones_genes_ruta)
    repeticiones_drogas_lista = cargar_repeticiones(repeticiones_drogas_ruta)
    print("repeticiones cargadas")

    embeddings_ruta = "E:/Descargas/Python/glove.6B.300d.txt"
    embeddings_dict = cargar_embeddings(embeddings_ruta)
    print("Embeddings cargados")

    publicaciones_directorio = "E:/Descargas/Python/PFC_DGIdb_src/scraping/files/txt"
    pmids_titulos_abstracts_keywords_ruta = "E:/Descargas/Python/PFC_DGIdb_src/scraping/pmids_titulos_abstracts_keywords.csv"
    publicaciones_dict = cargar_publicaciones(publicaciones_directorio,pmids_titulos_abstracts_keywords_ruta,pmids_lista)
    # print(publicaciones_dict)
    print("publicaciones cargadas")

    ocurrencias(aliases_gen_conjunto,publicaciones_dict,embeddings_dict,repeticiones_genes_lista,"ocurrencias_genes.csv")
    ocurrencias(aliases_droga_conjunto,publicaciones_dict,embeddings_dict,repeticiones_drogas_lista,"ocurrencias_drogas.csv")

    # aliases_repeticiones(aliases_gen_conjunto,aliases_gen_lista,"repeticiones_genes.csv")
    # aliases_repeticiones(aliases_droga_conjunto,aliases_droga_lista,"repeticiones_drogas.csv")

    

    # ifg_csv_ruta = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/pfc_dgidb_export_ifg.csv"
    # ifg_lista = cargar_ifg(ifg_csv_ruta)
    # print(ifg_lista)
    # print("Interacciones fármaco-gen cargadas")

    # etiquetas_publicacion_gen(pmids_lista,ifg_lista,aliases_gen_lista,"etiquetas_publicaciones_gen.csv")
    # etiquetas_publicacion_droga(pmids_lista,ifg_lista,aliases_droga_lista,"etiquetas_publicaciones_droga.csv")