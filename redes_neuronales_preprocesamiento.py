import os
import numpy as np
# from keras.preprocessing import text,sequence
import datos_preprocesamiento as dp
import csv
import re
import unittest

DIMENSION_EMBEDDING = 52

def ass(expression):
    if not expression: raise AssertionError()


class Test(unittest.TestCase):
    def test_uno(self):
        self.assertTrue(cargar_interacciones("test_emb"))


def cargar_ejemplos(etiquetas_neural_networks_ruta, ejemplos_directorio, out_interacciones_ruta,
                    incluir_sin_interacciones=False, top_palabras=20000, max_longitud=3000,
                    embeddings_file="glove.6B.50d.txt"):
    '''
    Carga los ejemplos para las redes neuronales en una lista de listas
    Entradas:
        etiquetas_neural_networks_ruta: archivo de interacciones fármaco-gen generadas
        ejemplos_directorio: directorio de los archivos txt de las publicaciones con los
                             remplazos hechos (xxx<nombre>xxx)
        out_interacciones: ruta del archivo con las interacciones de salida existentes, una por línea,
                           en el orden en que se mapeará la salida
    Salidas:
        xs: lista de matrices con los embeddings para servir como entrada a la red
        ys: lista de vectores de salida
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
                    data = ejemplo.read()
                    # lista = re.findall(r"\b\S+\b", data)
                    contenido_dict[pmid] = data
                    # if len(lista) > maxima_longitud:
                        # maxima_longitud = len(lista)
            if incluir_sin_interacciones or fila[3] != "sin_interaccion":
                pmids.append(pmid)
                genes.append(fila[1])
                drogas.append(fila[2])
                interacciones.append(fila[3])
    print("Listas pmids, genes, drogas, interacciones y diccionario de contenidos armados.")

    tokenizer = dp.Tokenizer(num_words=top_palabras)
    tokenizer.fit_on_texts(contenido_dict.values())
    print("Listo fit on texts.")
    # print("Top words:", [tokenizer.index_word[i] for i in range(1, top_palabras)])

    sum_longitud = 0
    for k in contenido_dict:
        v = contenido_dict[k]
        v = tokenizer.texts_to_top_words(v) # v queda como lista sólo con las top words
        contenido_dict[k] = v
        long = len(v)
        sum_longitud += long
        if long > maxima_longitud:
            maxima_longitud = long
    
    print("Longitud máxima:", maxima_longitud)
    print("Longitud promedio:", sum_longitud/len(contenido_dict))

    
    # hacer padding al inicio (llenar con ceros al inicio para que todos los ejemplos queden de la misma
    # longitud; para esto se agrega una palabra que no tenga embedding):
    for k in contenido_dict:
        v = contenido_dict[k]
        if len(v) < max_longitud:
            contenido_dict[k] = ["<PADDING>"] * (max_longitud - len(v)) + v
        else:
            contenido_dict[k] = v[:max_longitud]
        
        ass(len(contenido_dict[k]) == max_longitud)


    print("Listo padding.")

    embeddings_dict = dp.cargar_embeddings(embeddings_file)
    # DIMENSION_EMBEDDING = embeddings_dict[]
    interacciones_dict = cargar_interacciones(out_interacciones_ruta)

    # xs son las matrices de entrada; ys son los vectores de salida:
    xs = []; ys = []
    ll = len(pmids)
    for i in range(ll):
        print("Generando matrices: {}/{}".format(i+1, ll))
        contenido = contenido_dict[pmids[i]]
        ass(len(contenido) == max_longitud)
        x = generar_matriz_embeddings(contenido, genes[i], drogas[i], embeddings_dict)
        y = np.zeros((len(interacciones_dict)))
        y[interacciones_dict.get(interacciones[i], len(interacciones_dict)-1)] = 1
        xs.append(x)
        ys.append(y)
        
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    return xs, ys


def cargar_interacciones(in_file):
    """
    Carga el archivo de interacciones existentes de salida, y devuelve un diccionario
    con el índice de cada una para mapear el vector de salida. Por ejemplo:

        inhibitor
        agonist
        potentiator

    Cuando el caso de ejemplo sea de "inhibitor", el vector de salida será [1, 0, 0, 0].
    El último elemento se usa para agrupar las que no se nombran. Por ejemplo, con el listado
    anterior, cuando el caso de ejemplo sea de "modulator" el vector de salida será
    [0, 0, 0, 1].

    """
    with open(in_file, encoding="utf8") as f:
        res = {l.strip(): i for i, l in enumerate(f.readlines())}
    res["other"] = len(res)
    return res


def generar_matriz_embeddings(contenido, gen, droga, embeddings_dict):
    """
    Arma las matrices de entrada para la red. Funciona para un caso de ejemplo por vez.

    [p|p]      [x1|x2]
    [a|a]  ->  [x1|x2]
    [l|l]      [x1|x2]
    [1|2]      [x1|x2]
               [.....]

    Parámetros:
    embeddings_dict: diccionario de embeddings, de la forma [palabra] -> embedding (columna)
    """

    # print("Gen:", gen, " Droga:", droga)
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

    etiquetas_neural_networks_ruta = "etiquetas_neural_networks.csv"
    # etiquetas_neural_networks_ruta = "test_etiquetas"
    ejemplos_directorio = "replaced"
    # embeddings_ruta = "E:/Descargas/Python/glove.6B.300d.txt"
    out_interacciones_ruta = "interacciones_lista.txt"

    
    # test_file_ruta = "replaced/16534240.txt"
    # string = ""
    # with open(test_file_ruta,encoding="utf8") as test:
    #     string = test.read()
    # print(re.findall(r"\b\S+\b",string))

    cargar_ejemplos(etiquetas_neural_networks_ruta, ejemplos_directorio, out_interacciones_ruta)
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