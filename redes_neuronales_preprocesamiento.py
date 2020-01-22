import os
import numpy as np
import csv
import random
import math
import datos_preprocesamiento as dp
import funciones_auxiliares as ff

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
                    sin_interaccion_a_incluir=2944,
                    randomize=True,
                    porcentaje_test=0.1,
                    ejemplos_cantidad=-1):
    """
    Carga los ejemplos para las redes neuronales en una lista de listas
    Entradas:
        etiquetas_neural_networks_ruta:
            archivo de interacciones fármaco-gen generadas
        ejemplos_directorio:
            directorio de los archivos txt de las publicaciones con los
            remplazos hechos (xxx<nombre>xxx)
        out_interacciones_ruta:
            ruta del archivo con las interacciones de salida existentes, una por línea,
            en el orden en que se mapeará la salida
        incluir_sin_interacciones:
            determina si se cargan los ejemplos sintéticos de "sin_interacción"
        sin_interaccion_a_incluir:
            si se incluyen ejemplos "sin_interacción", determina cuántos se incluyen
        randomize:
            indica si se aleatoriza el orden de los ejemplos cargados
        porcentaje_test:
            porcentaje para separar un conjunto de datos para test
        ejemplos_cantidad:
            si se pasa un valor positivo, se usa esa cantidad de ejemplos para separar en
            entrenamiento y test balanceando las clases (hace oversampling o undersampling
            según necesite).
            Si es negativo usa el total que haya.
    Salidas:
        x_training: lista de matrices con los embeddings para servir como entrada a la red
        y_training: lista de vectores de salida
        x_test: lista de matrices del conjunto de test
        y_test: lista de vectores de salida correspondientes al conjunto de test
    """
    global DIMENSION_EMBEDDING
    etiquetas = []
    interacciones_sin = []
    contenido_dict = dict() # tiene un elemento por artículo: pmid -> contenido
    with open(etiquetas_neural_networks_ruta, encoding="utf8") as enn_csv:
        lector_csv = csv.reader(enn_csv, delimiter=',', quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            pmid = fila[0]
            if contenido_dict.get(pmid) is None:
                archivo_nombre = pmid + ".txt"
                archivo_ruta = os.path.join(ejemplos_directorio, archivo_nombre)
                with open(archivo_ruta, encoding="utf8") as ejemplo:
                    data = ejemplo.read()
                    contenido_dict[pmid] = data
            if incluir_sin_interacciones and fila[3] == "sin_interaccion":
                interacciones_sin.append(fila)
            elif incluir_sin_interacciones or fila[3] != "sin_interaccion":
                etiquetas.append(fila)

    interacciones_dict = cargar_interacciones(out_interacciones_ruta)

    # si se definió 'ejemplos_cantidad' se balancean las clases:
    if ejemplos_cantidad > 0:
        training, test = ff.balancear_clases(etiquetas_neural_networks_ruta,
                                             out_interacciones_ruta, ejemplos_cantidad,
                                             porcentaje_test)

    else: # sino se separa en test manteniendo proporción de clases:

        # tomar sólo x cantidad de sin_interaccion:
        if interacciones_sin:
            for fila in random.sample(interacciones_sin, k=sin_interaccion_a_incluir):
                etiquetas.append(fila)

        training, test = armar_test_set(etiquetas, interacciones_dict, porcentaje_test)

    print("Ejemplos (separados en training y test) y diccionario de contenidos armados.")
    print("Total para entrenamiento:", len(training), "Total para test:", len(test))

    tokenizer = dp.Tokenizer(num_words=top_palabras)
    tokenizer.fit_on_texts(contenido_dict.values())
    print("Listo fit on texts.")
    # print("Top words:", [tokenizer.index_word[i] for i in range(1, top_palabras)])
    
    embeddings_dict, maximo_valor_embedding, minimo_valor_embedding = dp.cargar_embeddings(embeddings_file)
    DIMENSION_EMBEDDING = len(next(iter(embeddings_dict.values())))

    x_training, y_training = _cargar_ejemplos(training, contenido_dict, tokenizer, max_longitud,
                                              embeddings_dict, maximo_valor_embedding, minimo_valor_embedding,
                                              interacciones_dict, randomize, "entrenamiento")
    x_test, y_test = _cargar_ejemplos(test, contenido_dict, tokenizer, max_longitud,
                                      embeddings_dict, maximo_valor_embedding, minimo_valor_embedding,
                                      interacciones_dict, randomize, "test")

    return (x_training, y_training), (x_test, y_test)


def _cargar_ejemplos(etiquetas, contenido_dict, tokenizer, max_longitud,
                     embeddings_dict, maximo_valor_embedding, minimo_valor_embedding,
                     interacciones_dict, randomize, tipo):
    # xs son las matrices de entrada; ys son los vectores de salida:
    ll = len(etiquetas)
    xs = np.empty((ll, max_longitud, DIMENSION_EMBEDDING))
    ys = np.empty((ll, len(interacciones_dict)))
    used_top_words = []
    for i in range(ll):
        pmid = etiquetas[i][0]
        gen = etiquetas[i][1]
        droga = etiquetas[i][2]
        interaccion = etiquetas[i][3]
        print("Generando matrices de {}: {}/{}".format(tipo, i+1, ll))

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
        xs[i] = x
        ys[i] = y

    # print("Promedio de top words usadas:", sum(used_top_words)/len(used_top_words))

    if randomize:
        # Aleatoriza el orden de los ejemplos
        print("Aleatorizando los ejemplos de {}...".format(tipo))
        seed = random.random()
        random.seed(seed)
        indices_aleatorios = np.arange(len(xs))
        random.shuffle(indices_aleatorios)
        xs = xs[indices_aleatorios]
        ys = ys[indices_aleatorios]
        print("Ejemplos de {} aleatorizados.".format(tipo))

    return xs, ys


def cargar_interacciones(in_file, invertir=False):
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
        if invertir:
            res = [l.strip() for l in f.readlines()]
        else:
            res = {l.strip(): i for i, l in enumerate(f.readlines())}
    # res["other"] = len(res)
    return res


def generar_matriz_embeddings(contenido, gen, droga, embeddings_dict, maximo_valor_embedding, minimo_valor_embedding):
    """
    Arma las matrices de entrada para la red. Funciona para un caso de ejemplo por vez.

    [p|a|l|1]  ->  [x1|x2|...]
    [p|a|l|2]      [x1|x2|...]

    Si se quiere intercambiar filas por columnas, usar x.transpose().

    Parámetros:
    embeddings_dict: diccionario de embeddings, de la forma [palabra] -> embedding (columna)
    """

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


def armar_test_set(etiquetas, interacciones_validas, porcentaje_test):
    """
    Toma 'porcentaje_test' de ejemplos de manera aleatoria de la lista de etiquetas,
    asegurándose de que casa clase (interacción) esté presente ese mismo porcentaje.
    Las interacciones tenidas en cuenta son las presentes en la lista 'interacciones_validas'
    (sino se cuentan como "other").

    Cada elemento de la lista de etiquetas es de la forma "[pmid, gen, droga, interacción]".
    El porcentaje de la forma 0.x.
    Devuelve dos listas: (training, test).
    """
    clases = {}
    test_set = []
    all_set = []
    for row in etiquetas:
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
    return all_set, test_set


if __name__ == "__main__":
    etiquetas_neural_networks_ruta = "etiquetas_neural_networks2.csv"
    ejemplos_directorio = "replaced"
    # etiquetas_neural_networks_ruta = "test_etiquetas"
    # ejemplos_directorio = "."
    # embeddings_ruta = "E:/Descargas/Python/glove.6B.300d.txt"
    out_interacciones_ruta = "interacciones_lista.txt"
    (xe, ye), (xt, yt) = cargar_ejemplos(etiquetas_neural_networks_ruta, ejemplos_directorio,
                                     out_interacciones_ruta, porcentaje_test=0.2,
                                     sin_interaccion_a_incluir=1)
    print("Cantidad x entrenamiento:", len(xe))
    print("Cantidad y entrenamiento:", len(ye))
    print("Cantidad x test:", len(xt))
    print("Cantidad y test:", len(yt))
