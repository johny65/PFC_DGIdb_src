import os
import numpy as np
# from keras.preprocessing import text,sequence
import datos_preprocesamiento as dp
import csv
import re

DIMENSION_EMBEDDING = 302


def cargar_ejemplos(etiquetas_neural_networks_ruta, ejemplos_directorio):
    '''
    Carga los ejemplos para las redes neuronales en una lista de listas
    Entradas:
        etiquetas_neural_networks_ruta: archivo de interacciones fármaco-gen generadas
        ejemplos_directorio: directorio de los archivos txt de las publicaciones con los remplazos hechos (xxx<nombre>xxx)
    Salidas:
        ejemplos_lista: lista con los ejemplos en formato lista de strings (ya separados).
        genes: lista de genes etiquetados.
        drogas: lista de drogas etiquetadas.
        maxima_longitud: longitud del ejemplo más largo.
    '''
    pmids = list()
    genes = list()
    drogas = list()
    interacciones = list()
    contenido_dict = dict() # tiene un elemento por artículo: pmid -> contenido
    maxima_longitud = 0
    with open(etiquetas_neural_networks_ruta, encoding="utf8") as enn_csv:
        lector_csv = csv.reader(enn_csv,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            pmid = fila[0]
            if contenido_dict.get(pmid) is None:
                # print("Agregando contenido de {} al diccionario".format(pmid))
                archivo_nombre = pmid + ".txt"
                archivo_ruta = os.path.join(ejemplos_directorio,archivo_nombre)
                with open(archivo_ruta,encoding="utf8") as ejemplo:
                    string = ejemplo.read()
                    lista = re.findall(r"\b\S+\b",string)
                    contenido_dict[pmid] = lista
                    if len(lista) > maxima_longitud:
                        maxima_longitud = len(lista)
            pmids.append(pmid)
            genes.append(fila[1])
            drogas.append(fila[2])
            interacciones.append(fila[3])
    print("Listas pmids,genes,drogas,interacciones y diccionario de contenidos armados.")
    
    # longitud; para esto se agrega una palabra que no tenga embedding):
    for k, v in contenido_dict.items():
        v = ["<PADDING>"] * (maxima_longitud - len(v)) + v
        assert len(v) == maxima_longitud

    embeddings_dict = dp.cargar_embeddings("glove.6B.300d.txt")

    # ejemplos_lista = list()
    for i in range(len(pmids)):
        contenido = contenido_dict[pmids[i]]
        em = generar_matriz_embeddings(contenido, genes[i], drogas[i], interacciones[i], embeddings_dict)
        print(em)
        # assert len(c) == maxima_longitud
        # ejemplos_lista.append(c)
        # print("Ejemplo {} con etiquetas {},{},{} cargado.".format(pmids[i],genes[i],drogas[i],interacciones[i]))
    # return ejemplos_lista,genes,drogas,maxima_longitud

def generar_matriz_embeddings(contenido, gen, droga, interaccion, embeddings_dict):
    """
    Arma las matrices de entrada para la red.

    [p|p]      [x1|x2]
    [a|a]  ->  [x1|x2]
    [l|l]      [x1|x2]
    [1|2]      [x1|x2]
               [.....]

    Parámetros:
    embeddings_dict: diccionario de embeddings, de la forma [palabra] -> embedding (columna)
    """

    print("Gen:", gen, " Droga:", droga, " Interacción:", interaccion)
    gen_emb = np.zeros((DIMENSION_EMBEDDING, 1))
    gen_emb[DIMENSION_EMBEDDING-2] = 1
    droga_emb = np.zeros((DIMENSION_EMBEDDING, 1))
    droga_emb[DIMENSION_EMBEDDING-1] = 1
    arreglo_base = np.zeros((DIMENSION_EMBEDDING, len(contenido)))

    for j in range(len(contenido)):
        palabra = contenido[j]
        if palabra == "xxx" + gen + "xxx":
            arreglo_base[:,j] = gen_emb[:,0]
        elif palabra == "xxx" + droga + "xxx":
            arreglo_base[:,j] = droga_emb[:,0]
        else:
            embedding_vector = embeddings_dict.get(palabra)
            if embedding_vector is not None:
                arreglo_base[:,j] = embedding_vector
    return arreglo_base

if __name__ == "__main__":

    # etiquetas_neural_networks_ruta = "etiquetas_neural_networks.csv"
    etiquetas_neural_networks_ruta = "test_etiquetas"
    ejemplos_directorio = "."
    # embeddings_ruta = "E:/Descargas/Python/glove.6B.300d.txt"

    
    # test_file_ruta = "replaced/16534240.txt"
    # string = ""
    # with open(test_file_ruta,encoding="utf8") as test:
    #     string = test.read()
    # print(re.findall(r"\b\S+\b",string))

    cargar_ejemplos(etiquetas_neural_networks_ruta, ejemplos_directorio)
    # ejemplos_lista,genes_lista,drogas_lista,maxima_longitud = cargar_ejemplos(etiquetas_neural_networks_ruta, ejemplos_directorio)
    # print("Ejemplos cargados.")
    # print(len(ejemplos_lista))
    # print(ejemplos_lista[226187])
    # print(maxima_longitud)
    # ejemplos_lista_arreglados,vocabulario = arreglar_ejemplos(ejemplos_lista)
    # print("Maxima longitud: {}".format(maxima_longitud))
    # print(ejemplos_lista_arreglados[0])
    # print(len(ejemplos_lista_arreglados[0]))
    # print(vocabulario.word_index["hello"])

    # embeddings_dict = datos_preprocesamiento.cargar_embeddings(embeddings_ruta)
    
    # print(len(embeddings_dict["hello"]))

    # a = [[1],[1,2],[1,2,3],[1],[1,2,3,4],[2,3,5],[1,2],[3,5]]
    # print(len(max(a,key=len)))

    # gen = np.zeros((1,10))
    # gen[0][-2] = 1
    # droga = np.zeros((1,10))
    # droga[0][-1] = 1
    # print(gen)
    # print(droga)

    # diccionario = dict()
    # diccionario["1"] = "hola1"
    # diccionario["2"] = "hola2"
    # diccionario["3"] = "hola3"
    # diccionario["4"] = "hola4"
    # diccionario["5"] = "hola5"
    # diccionario["6"] = "hola6"

    # print(diccionario["7"])
    # print(diccionario.get("7"))