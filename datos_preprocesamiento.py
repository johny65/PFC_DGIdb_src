import sys
import csv
import numpy as np
import re
import os
import logging
import preprocesamiento

logger = logging.getLogger("datos_preprocesamiento")

# Preprocesamiento de los datos (texto)
# from keras.preprocessing import text,sequence
# from preprocesamiento import remplazo_inteligente

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

def cargar_aliases_dict(aliases_ruta):
    '''
    Carga los aliases de gen o de droga dependiendo del archivo de entrada, en un diccionario
    de la forma: alias -> [nombres reales con ese alias] (set)
    '''
    aliases = {}
    with open(aliases_ruta, encoding="utf8") as aliases_csv:
        lector_csv = csv.reader(aliases_csv,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            nombre_real = fila[0]
            for alias in fila:
                aliases.setdefault(alias, set()).add(nombre_real)
    return aliases

def longitud_maxima_alias(aliases_ruta_lista):
    '''
    Entrada: lista con las rutas de los archivos de aliases.
    Salida: longitud del alias más largo.
    '''
    maxima_longitud = 0
    aliases_conjunto = set()
    for elemento in aliases_ruta_lista:
        aliases_conjunto = aliases_conjunto | cargar_aliases_conjunto(elemento)
    aliases_lista = list(aliases_conjunto)
    for elemento in aliases_lista:
        elemento_lista = elemento.split()
        longitud = len(elemento_lista)
        if longitud > maxima_longitud:
            maxima_longitud = longitud
    return maxima_longitud

def cargar_etiquetas_dict(ifg_file):
    '''
    Carga las etiquetas de las publicaciones en 2 diccionarios, uno para los genes y otro
    para las drogas. Cada diccionario tiene la forma: pmid -> listado de genes/drogas etiquetados.
    '''
    labels_genes = {}
    labels_drogas = {}
    with open(ifg_file, encoding="utf8") as f:
        lector_csv = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            pmid = fila[0]
            gen = fila[1]
            droga = fila[2]
            labels_genes.setdefault(pmid, set()).add(gen)
            labels_drogas.setdefault(pmid, set()).add(droga)
    return labels_genes, labels_drogas

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

def cargar_abstracts(pmids_titulos_abstracts_keywords_ruta):
    """Carga en un diccionario para cada pmid su abstract (pasado a minúscula)."""
    logger.debug("Cargando diccionario de abstracts...")
    abstracts_dict = {}
    with open(pmids_titulos_abstracts_keywords_ruta, encoding="utf8") as abstracts:
        lector_csv = csv.reader(abstracts,delimiter=',',quoting=csv.QUOTE_ALL)
        for linea in lector_csv:
            pmid = linea[0]
            abstract = linea[1]
            logger.debug("PMID: %s, abstract: len %d", pmid, len(abstract))
            abstracts_dict[pmid] = abstract.lower()
    return abstracts_dict

def cargar_pmids(pmids_file):
    """Carga de un archivo un listado de PMID. Cada línea del archivo debe ser de la forma
    [pmid algún texto]."""
    pmids = []
    with open(pmids_file, encoding="utf8") as f:
        for l in f:
            pmids.append(l.split()[0])
    return pmids

def cargar_publicaciones(publicaciones_directorio, abstracts_dict, pmids_lista):
    '''
    Carga las publicaciones que ya se encuentran en formato txt.
    Si la publicación no existe carga el titulo_abstract_keywords en su lugar.
    '''
    files_in_dir = sorted(os.listdir(publicaciones_directorio))
    publicaciones_dict = dict()
    for pmid in pmids_lista:
        logging.info(pmid)
        archivo_nombre = pmid + ".txt"
        if archivo_nombre in files_in_dir:
            archivo_ruta = os.path.join(publicaciones_directorio,archivo_nombre)
            with open(archivo_ruta,encoding="utf8") as publicacion:
                publicaciones_dict[pmid] = publicacion.read().lower()
        else:
            publicaciones_dict[pmid] = abstracts_dict[pmid]
    return publicaciones_dict

def ocurrencias(entidades,publicaciones,embeddings,repeticiones,salida_no_embedding_no_repeticiones): # ,salida_no_embedding_con_repeticiones,salida_todas
    '''
    Muestra las apariciones de genes/aliases o drogas/aliases en las publicaciones que no tienen embedding
    '''
    salida_no_embedding_no_repeticiones_csv = open(salida_no_embedding_no_repeticiones,"w",encoding="utf8")
    # salida_no_embedding_con_repeticiones_csv = open(salida_no_embedding_con_repeticiones,"w",encoding="utf8")
    # salida_todas_csv = open(salida_todas,"w",encoding="utf8")

    escritor_csv1 = csv.writer(salida_no_embedding_no_repeticiones_csv,delimiter=',',lineterminator="\n")
    # escritor_csv2 = csv.writer(salida_no_embedding_con_repeticiones_csv,delimiter=',',lineterminator="\n")
    # escritor_csv3 = csv.writer(salida_todas_csv,delimiter=',',lineterminator="\n")

    for elemento in publicaciones:
        print(elemento)
        lista1 = list()
        # lista2 = list()
        # lista3 = list()
        for entidad in entidades:
            if entidad in publicaciones[elemento] and entidad not in embeddings and entidad not in repeticiones:
                lista1.append(entidad)
            # if entidad in publicaciones[elemento] and entidad not in embeddings:
            #     lista2.append(entidad)
            # if entidad in publicaciones[elemento]:
            #     lista3.append(entidad)
        escritor_csv1.writerow(lista1)
        # escritor_csv2.writerow(lista2)
        # escritor_csv3.writerow(lista3)
    
    salida_no_embedding_no_repeticiones_csv.close()
    # salida_no_embedding_con_repeticiones_csv.close()
    # salida_todas_csv.close()

def cargar_ocurrencias(in_file):
    """A partir de un archivo de ocurrencias de genes/drogas crea un diccionario con
    ese listado para cada pmid."""
    ocs = {}
    with open(in_file, encoding="utf8") as f:
        lector_csv = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL)
        for linea in lector_csv:
            if linea:
                ocs[linea[0]] = linea[1:]
    return ocs

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

# -----------------------------------------

def cargar_publicaciones_con_remplazos(publicaciones_directorio):
    '''
    Carga las publicaciones que se encuentran en formato txt con los remplazos de genes y drogas hechos.
    '''
    archivos_en_directorio = sorted(os.listdir(publicaciones_directorio))
    publicaciones_dict = dict()
    for archivo in archivos_en_directorio:
        archivo_ruta = os.path.join(publicaciones_directorio,archivo)
        with open(archivo_ruta,encoding="utf8") as publicacion:
            publicaciones_dict[archivo.split(".")[0]] = publicacion.read()
    return publicaciones_dict

def cargar_nombres(alias_ruta):
    '''
    Carga los nombres (nombres posta, la primer columna) de genes y drogas en una lista
    '''
    nombres_lista = list()
    with open(alias_ruta,encoding="utf8") as aliases:
        lector_csv = csv.reader(aliases,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            nombres_lista.append(fila[0].lower())
    return nombres_lista

def ocurrencias_remplazos(publicaciones_dict,nombres_lista):
    '''
    Busca las ocurrencias de genes/drogas (según nombres_lista) que tengan el formato {[<nombre>]}.
    Las ocurrencias son guardadas en un diccionario con formato: pmid -> [<nombre1>, <nombre2>, ..., <nombreN>]
    '''
    ocurrencias_remplazos_dict = dict()
    for pmid,contenido in publicaciones_dict.items():
        # print("Buscando ocurrencias en {}.".format(pmid))
        nombres = list()
        for nombre in nombres_lista:
            nombre_clave = "{[" + nombre + "]}"
            if nombre_clave in contenido:
                print("Ocurrencia {} encontrada en {}.".format(nombre,pmid))
                nombres.append(nombre)
        if not nombres: # Si la lista nombres está vacía
            print("Ninguna ocurrencia encontrada en {}.".format(pmid))
        ocurrencias_remplazos_dict[pmid] = nombres
    return ocurrencias_remplazos_dict

def generar_etiquetas(ocurrencias_genes_dict,ocurrencias_drogas_dict,salida):
    '''
    Genera las etiquetas con todas las posibles combinaciones de genes y drogas con el formato: [pmid, gen, droga, "sin_interaccion"]
    Las etiquetas son guardadas en un archivo csv.
    '''
    with open(salida,"w",encoding="utf8") as salida_csv:
        escritor_csv = csv.writer(salida_csv,delimiter=',',lineterminator="\n")
        for pmid_gen,ocurrencias_gen in ocurrencias_genes_dict.items():
            for pmid_droga,ocurrencias_droga in ocurrencias_drogas_dict.items():
                if pmid_gen == pmid_droga:
                    print("Generando etiquetas para la publicación {}.".format(pmid_gen))
                    if not ((not ocurrencias_gen) or (not ocurrencias_droga)): # Solo cuando ambas listas tengan elementos
                        for gen in ocurrencias_gen:
                            for droga in ocurrencias_droga:
                                lista = [pmid_gen, gen, droga, "sin_interaccion"]
                                escritor_csv.writerow(lista)

def procesar_etiquetas(ifg_dgidb,ifg_generadas,salida):
    '''
    Se modifican las etiquetas generadas con la información de etiquetas correspondiente a DGIdb
    '''
    with open(salida,"w",encoding="utf8") as salida_csv:
        escritor_csv = csv.writer(salida_csv,delimiter=',',lineterminator="\n")
        for ifg_g in ifg_generadas:
            contador = 0
            for ifg_d in ifg_dgidb:
                if ifg_g[:-1] == ifg_d[:-1]:
                    print("Correspondencia con etiqueta de DGIdb encontrada.")
                    escritor_csv.writerow(ifg_d)
                    contador += 1
            if contador == 0:
                escritor_csv.writerow(ifg_g)

if __name__ == "__main__":
    # if len(sys.argv) != 1:
    #     print("Forma de uso: {} entrada salida".format(sys.argv[0]))
    #     exit()
    
    # pmids_etiquetas_completas_csv = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/pmids_etiquetas_completas.csv"
    # pmids_lista = cargar_pmids(pmids_etiquetas_completas_csv)
    # print(pmids_lista)
    # print("pmids cargados")

    aliases_gen_ruta = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/alias_gen.csv"
    # aliases_gen_conjunto = cargar_aliases_conjunto(aliases_gen_ruta)
    # aliases_gen_lista = cargar_aliases_lista(aliases_gen_ruta)
    # print(aliases_gen)
    # print("alias gen cargados")

    aliases_droga_ruta = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/alias_droga.csv"
    # aliases_droga_conjunto = cargar_aliases_conjunto(aliases_droga_ruta)
    # aliases_droga_lista = cargar_aliases_lista(aliases_droga_ruta)
    # print(aliases_droga)
    # print("alias droga cargados")

    # repeticiones_genes_ruta = "E:/Descargas/Python/PFC_DGIdb_src/repeticiones_genes.csv"
    # repeticiones_drogas_ruta = "E:/Descargas/Python/PFC_DGIdb_src/repeticiones_drogas.csv"
    # repeticiones_genes_lista = cargar_repeticiones(repeticiones_genes_ruta)
    # repeticiones_drogas_lista = cargar_repeticiones(repeticiones_drogas_ruta)
    # print("repeticiones cargadas")

    # embeddings_ruta = "E:/Descargas/Python/glove.6B.300d.txt"
    # embeddings_dict = cargar_embeddings(embeddings_ruta)
    # print("Embeddings cargados")

    # publicaciones_directorio = "E:/Descargas/Python/PFC_DGIdb_src/scraping/files/labeled/txt/txt_ungreek"
    # pmids_titulos_abstracts_keywords_ruta = "E:/Descargas/Python/PFC_DGIdb_src/scraping/pmids_titulos_abstracts_keywords.csv"
    # publicaciones_dict = cargar_publicaciones(publicaciones_directorio,pmids_titulos_abstracts_keywords_ruta,pmids_lista)
    # print(publicaciones_dict)
    # print("publicaciones cargadas")

    # ocurrencias(aliases_gen_conjunto,publicaciones_dict,embeddings_dict,repeticiones_genes_lista,"ocurrencias_genes_se_sr.csv") # ,"ocurrencias_genes_se_cr.csv","ocurrencias_genes_todas.csv"
    # ocurrencias(aliases_droga_conjunto,publicaciones_dict,embeddings_dict,repeticiones_drogas_lista,"ocurrencias_drogas_se_sr.csv") # ,"ocurrencias_drogas_se_cr.csv","ocurrencias_drogas_todas.csv"

    # aliases_repeticiones(aliases_gen_conjunto,aliases_gen_lista,"repeticiones_genes.csv")
    # aliases_repeticiones(aliases_droga_conjunto,aliases_droga_lista,"repeticiones_drogas.csv")

    # ifg_csv_ruta = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/pfc_dgidb_export_ifg.csv"
    # ifg_lista = cargar_ifg(ifg_csv_ruta)
    # print(ifg_lista)
    # print("Interacciones fármaco-gen cargadas")

    # etiquetas_publicacion_gen(pmids_lista,ifg_lista,aliases_gen_lista,"etiquetas_publicaciones_gen.csv")
    # etiquetas_publicacion_droga(pmids_lista,ifg_lista,aliases_droga_lista,"etiquetas_publicaciones_droga.csv")

    # archivo_ruta = "E:/Descargas/Python/PFC_DGIdb_src/scraping/files/labeled/txt/txt_ungreek/10722.txt"
    # f = open(archivo_ruta,encoding="utf8")
    # content = f.read()
    # print(content)
    # content_remplazado = preprocesamiento.remplazo_inteligente(content,"s t r u c t u r a l l y to the n a t u r a l catecholamines.","XXXXXXXXXXXXXXX")
    # s = re.sub("\w+\.",'\0',content)
    # print(content_remplazado)
    # f.close()
  
    # maxima_longitud = longitud_maxima_alias([aliases_gen_ruta, aliases_droga_ruta])
    # print(maxima_longitud)

    # lista = preprocesamiento.unir_elementos_lista(["a","b","c","d","e","f","g"],3,2)
    # print(lista)

    publicaciones_directorio = "E:/Descargas/Python/PFC_DGIdb_src/scraping/files/labeled/txt/txt_ungreek"
    publicaciones_dict = cargar_publicaciones_con_remplazos(publicaciones_directorio)
    print("Publicaciones cargadas.")

    nombres_gen_lista = cargar_nombres(aliases_gen_ruta)
    print("Nombres de genes cargados.")
    nombres_droga_lista = cargar_nombres(aliases_droga_ruta)
    print("Nombres de drogas cargados.")
    
    ocurrencias_remplazos_genes_dict = ocurrencias_remplazos(publicaciones_dict,nombres_gen_lista)
    print("Ocurrencias de remplazos de genes obtenidas.")
    ocurrencias_remplazos_drogas_dict = ocurrencias_remplazos(publicaciones_dict,nombres_droga_lista)
    print("Ocurrencias de remplazos de drogas obtenidas.")

    generar_etiquetas(ocurrencias_remplazos_genes_dict,ocurrencias_remplazos_drogas_dict,"etiquetas_generadas.csv")
    print("Generación de etiquetas completa")

    ifg_dgidb_ruta = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/pfc_dgidb_export_ifg.csv"
    ifg_dgidb = cargar_ifg(ifg_dgidb_ruta)
    print("Interacciones fármaco-gen de DGIdb cargadas.")

    ifg_generadas_ruta = "E:/Descargas/Python/PFC_DGIdb_src/etiquetas_generadas.csv"
    ifg_generadas = cargar_ifg(ifg_generadas_ruta)
    print("Interacciones fármaco-gen generadas cargadas.")
    
    procesar_etiquetas(ifg_dgidb,ifg_generadas,"etiquetas_neural_networks.csv")
    print("Procesamiento de etiquetas finalizado.")

    # a = [1,2,3,4,5,6,7,8,9,0]
    # for e in a[:-1]:
    #     print(e)

# if __name__ == "__main__":
# a = cargar_aliases_dict(sys.argv[1])
#     print(a["creatine kinase m chain"])
#     print(a["ec 3.4.24"])

    # eg, ed = cargar_etiquetas_dict("PFC_DGIdb/pfc_dgidb_export_ifg.csv")
    # print(eg["29133"])
    # print(ed["29133"])
    # print(eg["10722"])
    # print(ed["10722"])
    