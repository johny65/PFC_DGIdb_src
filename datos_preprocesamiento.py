import sys
import csv
import numpy as np
import re
import os
import logging
import tokenizer
from scraping import parallel
Tokenizer = tokenizer.Tokenizer

logging.basicConfig()
logger = logging.getLogger("datos_preprocesamiento")
logger.setLevel(logging.INFO)


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
                if e: aliases_conjunto.add(e)
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
    de la forma: alias -> [nombres reales con ese alias] (lista)
    '''
    aliases = {}
    with open(aliases_ruta, encoding="utf8") as aliases_csv:
        lector_csv = csv.reader(aliases_csv,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            nombre_real = fila[0]
            for alias in fila:
                if alias:
                    ya_alias = aliases.setdefault(alias, [])
                    if not alias in ya_alias:
                        ya_alias.append(nombre_real)
                else:
                    logger.warning("Para %s, nombre real %s se encontró un alias vacío", aliases_ruta, nombre_real)
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
    print("Cargando archivo de embeddings...")
    embeddings_dict = dict()
    maximo = 0.0
    minimo = 0.0
    with open(embeddings_ruta,encoding="utf8") as embeddings:
        for fila in embeddings:
            datos = fila.split()
            palabra = datos[0]
            embedding = np.asarray(datos[1:] + [0, 0], dtype='float64')
            embeddings_dict[palabra] = embedding
            maximo_vector = max(embedding)
            if maximo_vector > maximo:
                maximo = maximo_vector
            minimo_vector = min(embedding)
            if minimo_vector < minimo:
                minimo = minimo_vector
            # print("Embedding de la palabra <{}> cargado.".format(palabra))
    print("Carga de embeddings finalizada.")
    return embeddings_dict, maximo, minimo

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

def alias_repetidos(aliases_lista):
    """
    Busca los genes/drogas que tienen alias repetidos.
    aliases_conjunto: universo de entidades con sus alias.
    aliases_lista: cada elemento es una lista con una entidad y sus alias.
    """
    contador = {}
    for fila in aliases_lista:
        for e in fila:
            if e: # quedaron alias vacíos
                contador.setdefault(e, 0)
                contador[e] += 1
    res = [k for k,v in contador.items() if v > 1]
    return res

def cargar_abstracts(pmids_titulos_abstracts_keywords_ruta):
    """Carga en un diccionario para cada pmid su abstract (pasado a minúscula)."""
    logger.debug("Cargando diccionario de abstracts...")
    abstracts_dict = {}
    with open(pmids_titulos_abstracts_keywords_ruta, encoding="utf8") as abstracts:
        lector_csv = csv.reader(abstracts,delimiter=',',quoting=csv.QUOTE_ALL)
        for linea in lector_csv:
            pmid = linea[0]
            abstract = linea[1]
            abstracts_dict[pmid] = abstract
    return abstracts_dict

def cargar_pmids(pmids_file, sep=None):
    """Carga de un archivo un listado de PMID. Cada línea del archivo debe ser de la forma
    [pmid algún texto]."""
    pmids = []
    with open(pmids_file, encoding="utf8") as f:
        for l in f:
            pmids.append(l.split(sep)[0].strip())
    return pmids

def cargar_publicaciones(publicaciones_directorio, abstracts_dict=None, pmids_lista=None):
    '''
    Carga las publicaciones que ya se encuentran en formato txt.
    Si la publicación no existe carga el titulo_abstract_keywords en su lugar.
    Si se pasa una lista de PMID, sólo carga esos, sino todos los que se encuentren en el directorio.
    '''
    files_in_dir = os.listdir(publicaciones_directorio)
    publicaciones_dict = dict()
    if not pmids_lista:
        pmids_lista = (f.split(".")[0] for f in files_in_dir)
    for pmid in pmids_lista:
        archivo_nombre = pmid + ".txt"
        if archivo_nombre in files_in_dir:
            archivo_ruta = os.path.join(publicaciones_directorio, archivo_nombre)
            with open(archivo_ruta, encoding="utf8") as publicacion:
                publicaciones_dict[pmid] = publicacion.read()
        elif abstracts_dict:
            publicaciones_dict[pmid] = abstracts_dict[pmid]
    return publicaciones_dict

def ocurrencias(entidades, publicaciones, embeddings, repeticiones, tipo, index):
    """Busca las ocurrencias de las entidades en las publicaciones."""

    salida_no_embedding_no_repeticiones_csv = open("{}_se_sr_{}.csv".format(tipo, index), "w", encoding="utf8")
    salida_no_embedding_con_repeticiones_csv = open("{}_se_cr_{}.csv".format(tipo, index), "w", encoding="utf8")
    salida_todas_csv = open("{}_ce_cr_{}.csv".format(tipo, index), "w", encoding="utf8")

    escritor_csv1 = csv.writer(salida_no_embedding_no_repeticiones_csv,delimiter=',',lineterminator="\n")
    escritor_csv2 = csv.writer(salida_no_embedding_con_repeticiones_csv,delimiter=',',lineterminator="\n")
    escritor_csv3 = csv.writer(salida_todas_csv,delimiter=',',lineterminator="\n")

    for pmid, pub in publicaciones.items():
        lista1 = [pmid]
        lista2 = [pmid]
        lista3 = [pmid]
        for entidad in entidades:
            esta = re.search(r"\b{}\b".format(re.escape(entidad)), pub)
            # esta = entidad in pub
            if esta and entidad not in embeddings and entidad not in repeticiones:
                logger.debug("Entidad '%s' no tiene embedding ni repetición.", entidad)
                lista1.append(entidad)
            if esta and entidad not in embeddings:
                logger.debug("Entidad '%s' no tiene embedding.", entidad)
                lista2.append(entidad)
            if esta:
                logger.debug("Entidad '%s' tiene embedding y repetición.", entidad)
                lista3.append(entidad)
        escritor_csv1.writerow(lista1)
        escritor_csv2.writerow(lista2)
        escritor_csv3.writerow(lista3)
    
    salida_no_embedding_no_repeticiones_csv.close()
    salida_no_embedding_con_repeticiones_csv.close()
    salida_todas_csv.close()
    return lista1, lista2, lista3

def cargar_ocurrencias(in_file, ocs=None):
    """A partir de un archivo de ocurrencias de genes/drogas crea un diccionario con
    ese listado para cada pmid."""
    ocs = ocs or {}
    with open(in_file, encoding="utf8") as f:
        lector_csv = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL)
        for linea in lector_csv:
            if linea:
                ocs[linea[0]] = linea[1:] if len(linea) > 1 else []
    return ocs

def sumar_ocurrencias(existentes_file, nuevas_file):
    """Suma nuevas ocurrencias a un archivo existente."""
    ocs = cargar_ocurrencias(existentes_file)
    ocs = cargar_ocurrencias(nuevas_file, ocs)
    # ordeno por pmid:
    with open(existentes_file, "w") as out:
        writer = csv.writer(out)
        writer.writerows([[k] + v for k, v in sorted(ocs.items(), key=lambda i: int(i[0]))])


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

# Parte dedicada a la generación de las etiquetas

def cargar_pmids_genes_drogas_unicos(dgidb_export_ifg_csv):
    '''
    Carga los pmids con las etiquetas completas en una lista ordenada y sin repeticiones.
    Los elementos son de tipo entero.
    '''
    pmids_conjunto = set()
    genes_conjunto = set()
    drogas_conjunto = set()
    pmids_lista = list()
    genes_lista = list()
    drogas_lista = list()
    with open(dgidb_export_ifg_csv,encoding="utf8") as pmids_csv:
        lector_csv = csv.reader(pmids_csv,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            pmids_conjunto.add(int(fila[0]))
            genes_conjunto.add(fila[1])
            drogas_conjunto.add(fila[2])
        pmids_lista = list(pmids_conjunto)
        genes_lista = list(genes_conjunto)
        drogas_lista = list(drogas_conjunto)
        pmids_lista.sort()
        genes_lista.sort()
        drogas_lista.sort()
    return pmids_lista, genes_lista, drogas_lista

def cargar_entidades_etiquetas_dgidb(pmids_lista,ifg_dgidb_lista):
    '''
    A partir de una lista de pmids (números enteros) y una lista de lista con las interacciones fármaco-gen,
    devuelve dos diccionarios con el formato:
        pmid -> [<gen1>, <gen2>, ..., <genN>]
        pmid -> [<droga1>, <droga2>, ..., <drogaN>]
    conteniendo las entidades nombradas en las etiquetas sin repeticiones.
    '''
    genes_dict = dict()
    drogas_dict = dict()
    for pmid in pmids_lista:
        pmid = str(pmid)
        genes_conjunto = set()
        drogas_conjunto = set()
        for lista in ifg_dgidb_lista:
            if pmid == lista[0]:
                genes_conjunto.add(lista[1])
                drogas_conjunto.add(lista[2])
        genes_dict[pmid] = list(genes_conjunto)
        drogas_dict[pmid] = list(drogas_conjunto)
    return genes_dict, drogas_dict

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
    Carga los nombres (nombres posta, la primer columna) de genes o drogas en una lista
    '''
    nombres_lista = list()
    with open(alias_ruta,encoding="utf8") as aliases:
        lector_csv = csv.reader(aliases,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            nombres_lista.append(fila[0].lower())
    return nombres_lista

def ocurrencias_remplazos(publicaciones_dict,nombres_lista):
    '''
    Busca las ocurrencias de genes/drogas (según nombres_lista) que tengan el formato xxx<nombre>xxx.
    Las ocurrencias son guardadas en un diccionario con formato: pmid -> [<nombre1>, <nombre2>, ..., <nombreN>]
    '''
    ocurrencias_remplazos_dict = dict()
    for pmid,contenido in publicaciones_dict.items():
        # print("Buscando ocurrencias en {}.".format(pmid))
        nombres = list()
        for nombre in nombres_lista:
            nombre_clave = "xxx" + nombre + "xxx"
            if nombre_clave in contenido:
                # print("Ocurrencia {} encontrada en {}.".format(nombre,pmid))
                nombres.append(nombre)
        # if not nombres: # Si la lista nombres está vacía
            # print("Ninguna ocurrencia encontrada en {}.".format(pmid))
        # print("Ocurrencias encontradas en {}: {}".format(pmid,nombres))
        ocurrencias_remplazos_dict[pmid] = nombres
    return ocurrencias_remplazos_dict


def generar_etiquetas(ocurrencias_genes_dict, # pmid -> [genes encontrados en ese pmid]
                    ocurrencias_drogas_dict,  # pmid -> [drogas encontradas en ese pmid]
                    genes_etiquetados_dict,   # pmid -> [genes en ese pmid que están etiquetados en algún ejemplo]
                    drogas_etiquetadas_dict,  # pmid -> [drogas en ese pmid que están etiquetadas en algún ejemplo]
                    genes_lista,              # lista global de genes etiquetados
                    drogas_lista,             # lista global de drogas etiquetadas
                    salida):                  # ruta para el archivo de salida
    '''
    Genera las etiquetas con todas las posibles combinaciones de genes y drogas con el formato: [pmid, gen, droga, "sin_interaccion"]
    Las etiquetas son guardadas en un archivo csv.
    '''
    # with open(salida,"w",encoding="utf8") as salida_csv:
    #     escritor_csv = csv.writer(salida_csv,delimiter=',',lineterminator="\n")
    #     for pmid_gen,ocurrencias_gen in ocurrencias_genes_dict.items():
    #         for pmid_droga,ocurrencias_droga in ocurrencias_drogas_dict.items():
    #             if pmid_gen == pmid_droga:
    #                 print("Generando etiquetas para la publicación {}.".format(pmid_gen))
    #                 if not ((not ocurrencias_gen) or (not ocurrencias_droga)): # Solo cuando ambas listas tengan elementos
    #                     agregados = list()
    #                     for gen in ocurrencias_gen:
    #                         genes_e = genes_etiquetados_dict[pmid_gen]
    #                         if gen in genes_e:
    #                             for droga in ocurrencias_droga:
    #                                 if droga in drogas_lista:
    #                                     lista = [pmid_gen, gen, droga, "sin_interaccion"]
    #                                     agregados.append(lista)
    #                                     escritor_csv.writerow(lista)
    #                     for droga in ocurrencias_droga:
    #                         drogas_e = drogas_etiquetadas_dict[pmid_droga]
    #                         if droga in drogas_e and droga in drogas_lista:
    #                             for gen in ocurrencias_gen:
    #                                 if gen in genes_lista:
    #                                     lista = [pmid_droga, gen, droga, "sin_interaccion"]
    #                                     if lista not in agregados:
    #                                         escritor_csv.writerow(lista)
    with open(salida,"w",encoding="utf8") as salida_csv:
        escritor_csv = csv.writer(salida_csv,delimiter=',',lineterminator="\n")
        for pmid_gen,ocurrencias_gen in ocurrencias_genes_dict.items():
            for pmid_droga,ocurrencias_droga in ocurrencias_drogas_dict.items():
                if pmid_gen == pmid_droga:
                    print("Generando etiquetas para la publicación {}.".format(pmid_gen))
                    if ocurrencias_gen and ocurrencias_droga: # Solo cuando ambas listas tengan elementos
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
                    print("Correspondencia con etiqueta de DGIdb encontrada: {},{},{}".format(ifg_g[0],ifg_g[1],ifg_g[2]))
                    escritor_csv.writerow(ifg_d)
                    contador += 1
            if contador == 0:
                escritor_csv.writerow(ifg_g)


def main_generar_ocurrencias():
    """Para generar los archivos de ocurrencias."""

    pmids_etiquetas_completas_csv = "PFC_DGIdb/pmids_etiquetas_completas.csv"
    pmids_lista = cargar_pmids(pmids_etiquetas_completas_csv)
    print("pmids cargados")

    aliases_gen_ruta = "PFC_DGIdb/alias_gen.csv"
    aliases_gen_conjunto = cargar_aliases_conjunto(aliases_gen_ruta)
    aliases_gen_lista = cargar_aliases_lista(aliases_gen_ruta)
    print("alias gen cargados")

    aliases_droga_ruta = "PFC_DGIdb/alias_droga.csv"
    aliases_droga_conjunto = cargar_aliases_conjunto(aliases_droga_ruta)
    aliases_droga_lista = cargar_aliases_lista(aliases_droga_ruta)
    print("alias droga cargados")

    repeticiones_genes_lista = alias_repetidos(aliases_gen_lista)
    repeticiones_drogas_lista = alias_repetidos(aliases_droga_lista)
    print("repeticiones cargadas")

    embeddings_ruta = "glove.6B.50d.txt"
    embeddings_dict = cargar_embeddings(embeddings_ruta)[0] # devuelve 3 elementos
    print("Embeddings cargados")

    pmids_titulos_abstracts_keywords_ruta = "scraping/pmids_titulos_abstracts_keywords.csv"
    abstracts_dict = cargar_abstracts(pmids_titulos_abstracts_keywords_ruta)
    print("Abstracts/títulos/palabras clave cargadas")

    publicaciones_directorio = "scraping/files/labeled/txt_ungreek"
    publicaciones_dict = cargar_publicaciones(publicaciones_directorio, abstracts_dict, pmids_lista)
    print("publicaciones cargadas:", len(publicaciones_dict))

    print("Buscando ocurrencias de genes...")
    parallel.parallel_map2(g_occs, publicaciones_dict, (aliases_gen_conjunto, embeddings_dict, repeticiones_genes_lista))
    
    print("Buscando ocurrencias de drogas...")
    parallel.parallel_map2(d_occs, publicaciones_dict, (aliases_droga_conjunto, embeddings_dict, repeticiones_drogas_lista))
    
    print("Listo, se guardaron en archivos.")


def g_occs(pubs, index, params):
    """Para paralelizar ocurrencias de genes."""
    aliases_gen_conjunto, embeddings_dict, repeticiones_genes_lista = params
    ocurrencias(aliases_gen_conjunto, pubs, embeddings_dict, repeticiones_genes_lista, "g", index)

def d_occs(pubs, index, params):
    """Para paralelizar ocurrencias de drogas."""
    aliases_droga_conjunto, embeddings_dict, repeticiones_drogas_lista = params
    ocurrencias(aliases_droga_conjunto, pubs, embeddings_dict, repeticiones_drogas_lista, "d", index)


def main_generar_etiquetas():
    publicaciones_directorio = "replaced4"
    publicaciones_dict = cargar_publicaciones_con_remplazos(publicaciones_directorio)
    print("Publicaciones cargadas.")

    ifg_dgidb_ruta = "PFC_DGIdb/pfc_dgidb_export_ifg.csv"
    pmids_lista, genes_lista, drogas_lista = cargar_pmids_genes_drogas_unicos(ifg_dgidb_ruta)
    print("Lista de pmids, genes y drogas cargadas.")

    ifg_dgidb = cargar_ifg(ifg_dgidb_ruta)
    print("Interacciones fármaco-gen de DGIdb cargadas.")

    genes_dict,drogas_dict = cargar_entidades_etiquetas_dgidb(pmids_lista,ifg_dgidb)
    print("Genes en etiquetas cargados")
    print("Drogas en etiquetas cargadas")

    aliases_gen_ruta = "PFC_DGIdb/alias_gen.csv"
    nombres_gen_lista = cargar_nombres(aliases_gen_ruta)
    print("Nombres de genes cargados.")
    
    aliases_droga_ruta = "PFC_DGIdb/alias_droga.csv"
    nombres_droga_lista = cargar_nombres(aliases_droga_ruta)
    print("Nombres de drogas cargados.")
    
    ocurrencias_remplazos_genes_dict = ocurrencias_remplazos(publicaciones_dict,nombres_gen_lista)
    print("Ocurrencias de remplazos de genes obtenidas.")
    ocurrencias_remplazos_drogas_dict = ocurrencias_remplazos(publicaciones_dict,nombres_droga_lista)
    print("Ocurrencias de remplazos de drogas obtenidas.")

    generar_etiquetas(ocurrencias_remplazos_genes_dict,ocurrencias_remplazos_drogas_dict,genes_dict,drogas_dict,genes_lista,drogas_lista,"etiquetas_generadas_aux.csv")
    print("Generación de etiquetas completa.")

    ifg_generadas_ruta = "etiquetas_generadas_aux.csv"
    ifg_generadas = cargar_ifg(ifg_generadas_ruta)
    print("Interacciones fármaco-gen generadas cargadas.")
    
    procesar_etiquetas(ifg_dgidb,ifg_generadas, "etiquetas_neural_networks_4_v2.csv")
    print("Procesamiento de etiquetas finalizado.")


if __name__ == "__main__":
    # main_generar_ocurrencias()
    main_generar_etiquetas()
