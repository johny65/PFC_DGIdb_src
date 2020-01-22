import os
import numpy as np
import datos_preprocesamiento as dp
import csv
import re
import unittest
import random
import math

DIMENSION_EMBEDDING = -1 # se autocalcula

def contar_top_words(strings_lista):
    contador = 0
    for string in strings_lista:
        if not (string.startswith("xxx") and string.endswith("xxx")):
            contador += 1
    return contador

def ass(expression):
    if not expression: raise AssertionError()


def cargar_ejemplos(etiquetas_neural_networks_ruta,
                    ejemplos_directorio,
                    out_interacciones_ruta,
                    incluir_sin_interacciones=True,
                    top_palabras=150,
                    max_longitud=500,
                    embeddings_file="glove.6B.50d.txt",
                    sin_interaccion_a_incluir=2944):
                    # agregar PROCENTAJE_PRUEBA
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
    global DIMENSION_EMBEDDING
    etiquetas = []
    interacciones_sin = []
    contenido_dict = dict() # tiene un elemento por artículo: pmid -> contenido
    with open(etiquetas_neural_networks_ruta, encoding="utf8") as enn_csv:
        lector_csv = csv.reader(enn_csv, delimiter = ',', quoting = csv.QUOTE_ALL)
        for fila in lector_csv:
            pmid = fila[0]
            if contenido_dict.get(pmid) is None:
                archivo_nombre = pmid + ".txt"
                archivo_ruta = os.path.join(ejemplos_directorio,archivo_nombre)
                with open(archivo_ruta,encoding="utf8") as ejemplo:
                    data = ejemplo.read()
                    contenido_dict[pmid] = data
            if incluir_sin_interacciones and fila[3] == "sin_interaccion":
                interacciones_sin.append(fila)
            elif incluir_sin_interacciones or fila[3] != "sin_interaccion":
                etiquetas.append(fila)

    # tomar sólo x cantidad de sin_interaccion:
    if interacciones_sin:
        for fila in random.sample(interacciones_sin, k=sin_interaccion_a_incluir):
        # for fila in interacciones_sin[:sin_interaccion_a_incluir]:
            etiquetas.append(fila)
            # pmids.append(pmid)
            # genes.append(fila[1])
            # drogas.append(fila[2])
            # interacciones.append(fila[3])

    print("Listas pmids, genes, drogas, interacciones y diccionario de contenidos armados.")

    tokenizer = dp.Tokenizer(num_words=top_palabras)
    tokenizer.fit_on_texts(contenido_dict.values())
    print("Listo fit on texts.")
    # print("Top words:", [tokenizer.index_word[i] for i in range(1, top_palabras)])
    
    embeddings_dict, maximo_valor_embedding, minimo_valor_embedding = dp.cargar_embeddings(embeddings_file)
    DIMENSION_EMBEDDING = len(next(iter(embeddings_dict.values())))
    interacciones_dict = cargar_interacciones(out_interacciones_ruta)

    # xs son las matrices de entrada; ys son los vectores de salida:
    ll = len(etiquetas)
    # xs = np.empty((ll, max_longitud, DIMENSION_EMBEDDING))
    # ys = np.empty((ll, len(interacciones_dict)))
    xs = []; ys = []
    used_top_words = []
    for i in range(ll):
        pmid = etiquetas[0]
        gen = etiquetas[1]
        droga = etiquetas[2]
        interaccion = etiquetas[3]

        print("Generando matrices: {}/{}".format(i+1, ll))
        contenido = contenido_dict[pmid]

        # contenido queda como lista sólo con las top words, y padeado en caso de ser necesario
        contenido = tokenizer.texts_to_top_words(contenido, max_longitud, gen, droga)
        used_top_words.append(tokenizer.used_top_words)

        if len(contenido) < max_longitud:
            # hacer padding al inicio (llenar con ceros al inicio para que todos los ejemplos queden de la misma
            # longitud; para esto se agrega una palabra que no tenga embedding):
            contenido = ["<PADDING>"] * (max_longitud - len(contenido)) + contenido
        else:
            contenido = contenido[:max_longitud]

        x = generar_matriz_embeddings(contenido, gen, droga, embeddings_dict, maximo_valor_embedding, minimo_valor_embedding)
        y = np.zeros((len(interacciones_dict)))
        y[interacciones_dict.get(interaccion, len(interacciones_dict)-1)] = 1
        # xs[i] = x
        # ys[i] = y
        xs.append(x)
        ys.append(y)

    print("Promedio de top words usadas:", sum(used_top_words)/len(used_top_words))
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    # Aleatoriza el orden de los ejemplos
    # seed = random.random()
    # random.seed(seed)
    # random.shuffle(xs)
    # random.seed(seed)
    # random.shuffle(ys)

    # Aleatoriza el orden de los ejemplos
    print("Aleatorizando los ejemplos de entrenamiento.")
    seed = random.random()
    random.seed(seed)
    indices_aleatorios = np.arange(len(xs))
    random.shuffle(indices_aleatorios)
    xs = xs[indices_aleatorios]
    ys = ys[indices_aleatorios]
    print("Ejemplos de entrenamiento aleatorizados.")

    return (xs, ys) #, (np.array([]), np.array([])) # (x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba)


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


def generar_matriz_embeddings(contenido, gen, droga, embeddings_dict, maximo_valor_embedding, minimo_valor_embedding):
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
    # gen_emb = np.zeros((DIMENSION_EMBEDDING, 1))
    # gen_emb[DIMENSION_EMBEDDING-2] = 1
    # droga_emb = np.zeros((DIMENSION_EMBEDDING, 1))
    # droga_emb[DIMENSION_EMBEDDING-1] = 1
    # arreglo_base = np.zeros((DIMENSION_EMBEDDING, len(contenido)))

    # Modificación: intercambio de filas por columnas para usar el Conv1D y el MaxPooling1D
    gen_emb = np.zeros((1, DIMENSION_EMBEDDING))
    gen_emb[0, DIMENSION_EMBEDDING-2] = 1
    droga_emb = np.zeros((1, DIMENSION_EMBEDDING))
    droga_emb[0, DIMENSION_EMBEDDING-1] = 1
    arreglo_base = np.zeros((len(contenido), DIMENSION_EMBEDDING))

    # Modificación de los vectores de genes y drogas
    # gen solo tiene cero en la posición de droga
    # droga solo tiene cero en la posición de gen
    # gen_emb = np.ones((DIMENSION_EMBEDDING, 1))
    # gen_emb[DIMENSION_EMBEDDING - 1] = 0
    # droga_emb = np.ones((DIMENSION_EMBEDDING, 1))
    # droga_emb[DIMENSION_EMBEDDING - 2] = 0

    # for j in range(len(contenido)):
    #     palabra = contenido[j]
    #     if palabra == "xxx" + gen + "xxx":
    #         arreglo_base[:,j] = gen_emb[:,0]
    #     elif palabra == "xxx" + droga + "xxx":
    #         arreglo_base[:,j] = droga_emb[:,0]
    #     else:
    #         embedding_vector = embeddings_dict.get(palabra)
    #         if embedding_vector is not None:
    #             arreglo_base[:,j] = embedding_vector # - minimo_valor_embedding) / (maximo_valor_embedding - minimo_valor_embedding) # normaliza los embeddings entre 0 y 1
    # return arreglo_base

    # Modificación: intercambio de filas por columnas para usar el Conv1D y el MaxPooling1D
    for j in range(len(contenido)):
        palabra = contenido[j]
        if palabra == "xxx" + gen + "xxx":
            arreglo_base[j,:] = gen_emb[0,:]
        elif palabra == "xxx" + droga + "xxx":
            arreglo_base[j,:] = droga_emb[0,:]
        else:
            embedding_vector = embeddings_dict.get(palabra)
            if embedding_vector is not None:
                arreglo_base[j,:] = embedding_vector # - minimo_valor_embedding) / (maximo_valor_embedding - minimo_valor_embedding) # normaliza los embeddings entre 0 y 1
    return arreglo_base


def armar_test_set(archivo_etiquetas, archivo_interacciones, porcentaje_test):
    """
    Toma 'porcentaje_test' de ejemplos de manera aleatoria del archivo de etiquetas,
    asegurándose de que casa clase (interacción) esté presente ese mismo porcentaje.
    Las interacciones tenidas en cuenta son las presentes en el archivo 'archivo_interacciones'
    (sino se cuentan como "other").

    Cada línea del archivo es de la forma "pmid,gen,droga,interacción".
    El porcentaje de la forma 0.x.
    Devuelve dos listas: los ejemplos separados para test, y el total sin esos ejemplos.
    El orden del conjunto de test queda aleatorio también.
    """
    interacciones_validas = cargar_interacciones(archivo_interacciones)
    clases = {}
    test_set = []
    all_set = []
    with open(archivo_etiquetas, encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            interaccion = row[3]
            if not interaccion in interacciones_validas:
                interaccion = "other"
            clases.setdefault(interaccion, []).append(row)
            all_set.append(row)
    for interaccion, ejemplos in clases.items():
        cantidad = len(ejemplos)
        # siempre deja al menos un elemento de cada clase (con el ceil)
        test_set += random.sample(ejemplos, k=math.ceil(cantidad*porcentaje_test))
    for t in test_set:
        all_set.remove(t)
    random.shuffle(test_set)
    return test_set, all_set


if __name__ == "__main__":
    etiquetas_neural_networks_ruta = "etiquetas_neural_networks2.csv"
    ejemplos_directorio = "replaced"
    # etiquetas_neural_networks_ruta = "test_etiquetas"
    # ejemplos_directorio = "."
    # embeddings_ruta = "E:/Descargas/Python/glove.6B.300d.txt"
    out_interacciones_ruta = "interacciones_lista.txt"
    cargar_ejemplos(etiquetas_neural_networks_ruta, ejemplos_directorio, out_interacciones_ruta)
    
    # t, a = armar_test_set(etiquetas_neural_networks_ruta, out_interacciones_ruta, 0.2)
    # print("Cantidad en todos:", len(a))
    # print("Cantidad en test:", len(t))
