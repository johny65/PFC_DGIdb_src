import os
import numpy as np
import csv
import random
import math
import datos_preprocesamiento as dp
import funciones_auxiliares as ff
from keras.preprocessing import text, sequence
from keras.utils import np_utils
import pickle

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
                    ejemplos_cantidad=-1,
                    vocabulario_global=True):
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
        vocabulario_global:
            Si es true se analizan todos los textos y se calculan las top words en base a todos.
            Si es false se calculan las top words por cada artículo.
    Salidas:
        x_training: lista de matrices con los embeddings para servir como entrada a la red
        y_training: lista de vectores de salida
        x_test: lista de matrices del conjunto de test
        y_test: lista de vectores de salida correspondientes al conjunto de test
    """
    global DIMENSION_EMBEDDING
    etiquetas = []
    interacciones_sin = []
    with open(etiquetas_neural_networks_ruta, encoding="utf8") as enn_csv:
        lector_csv = csv.reader(enn_csv, delimiter=',', quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            if incluir_sin_interacciones and fila[3] == "sin_interaccion":
                interacciones_sin.append(fila)
            elif incluir_sin_interacciones or fila[3] != "sin_interaccion":
                etiquetas.append(fila)

    contenido_dict = dp.cargar_publicaciones(ejemplos_directorio) # tiene un elemento por artículo: pmid -> contenido
    interacciones_dict = cargar_interacciones(out_interacciones_ruta)

    # si se definió 'ejemplos_cantidad' se balancean las clases:
    if ejemplos_cantidad > 0:
        training, test = ff.balancear_clases(etiquetas_neural_networks_ruta,
                                             out_interacciones_ruta, [], ejemplos_cantidad,
                                             porcentaje_test)

    else: # sino se separa en test manteniendo proporción de clases:

        # tomar sólo x cantidad de sin_interaccion:
        if interacciones_sin:
            for fila in random.sample(interacciones_sin, k=sin_interaccion_a_incluir):
                etiquetas.append(fila)

        training, test = armar_test_set(etiquetas, interacciones_dict, porcentaje_test)

    print("Ejemplos (separados en training y test) y diccionario de contenidos armados.")
    print("Total para entrenamiento:", len(training), "Total para test:", len(test))

    if vocabulario_global:
        tokenizer = dp.Tokenizer(num_words=top_palabras)
        tokenizer.fit_on_texts(contenido_dict.values())
        print("Listo fit on texts.")
        # print("Top words:", [tokenizer.index_word[i] for i in range(1, top_palabras)])
    else:
        tokenizer = None
    
    embeddings_dict, maximo_valor_embedding, minimo_valor_embedding = dp.cargar_embeddings(embeddings_file)
    DIMENSION_EMBEDDING = len(next(iter(embeddings_dict.values())))

    x_training, y_training = _cargar_ejemplos(training, contenido_dict, tokenizer, top_palabras, max_longitud,
                                              embeddings_dict, maximo_valor_embedding, minimo_valor_embedding,
                                              interacciones_dict, randomize, "entrenamiento")
    x_test, y_test = _cargar_ejemplos(test, contenido_dict, tokenizer, top_palabras, max_longitud,
                                      embeddings_dict, maximo_valor_embedding, minimo_valor_embedding,
                                      interacciones_dict, randomize, "test")

    return (x_training, y_training), (x_test, y_test)


def _cargar_ejemplos(etiquetas, contenido_dict, tokenizer, top_palabras, max_longitud,
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
        if not tokenizer:
            tokenizer = dp.Tokenizer(num_words=top_palabras)
            tokenizer.fit_on_texts([contenido])
        contenido = tokenizer.texts_to_top_words(contenido, max_longitud, gen, droga)
        used_top_words.append(tokenizer.used_top_words)

        # Para mostrar las top words por ejemplo y el contenido de los mismos
        # print("TOP WORDS:")
        # print("Top words:", [tokenizer.index_word[i] for i in range(1, top_palabras)])
        # print("CONTENIDO:")
        # print(contenido)

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
        # seed = random.random()
        # random.seed(seed)
        # random.shuffle(xs)
        # random.seed(seed)
        # random.shuffle(ys)

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

# ------------------------------------------------------------------------------------------

def compactar_lista(lista):
    '''
    Compacta elementos iguales consecutivos.
    '''
    elemento_anterior = ""
    lista_compactada = list()
    for elemento in lista:
        if elemento != elemento_anterior:
            lista_compactada.append(elemento)
            elemento_anterior = elemento
    return lista_compactada

def acortar_secuencia(secuencia_lista, vocabulario, ifg_balanceadas_lista, maxima_longitud_ejemplos):
    suma = 0
    for i in range(0, len(secuencia_lista), 1):
        # print("Reduciendo el ejemplo: {}".format(ifg_balanceadas_lista[i]))
        # print(len(secuencia_lista[i]))

        # no hace más falta llamar al compactar, este algoritmo ya evita que queden repetidos secuenciales
        ejemplo = secuencia_lista[i]
        suma += len(ejemplo)
        gen = ifg_balanceadas_lista[i][1]
        droga = ifg_balanceadas_lista[i][2]
        
        num_words = maxima_longitud_ejemplos
        ready = False
        while not ready:
            ejemplo_reducido = list()
            for elemento in ejemplo:
                palabra = vocabulario.index_word[elemento]
                if palabra.startswith("xxx") and (palabra == "xxx{}xxx".format(gen) or palabra == "xxx{}xxx".format(droga)):
                    # gen o droga de interés
                    if not ejemplo_reducido or (ejemplo_reducido and ejemplo_reducido[-1] != elemento):
                        ejemplo_reducido.append(elemento)
                elif elemento <= num_words:
                    # agregar sólo si está dentro de máxima longitud
                    if not ejemplo_reducido or (ejemplo_reducido and ejemplo_reducido[-1] != elemento):
                        ejemplo_reducido.append(elemento)
            if len(ejemplo_reducido) <= maxima_longitud_ejemplos or num_words < 0:
                ready = True
            else:
                num_words -= 1
                ejemplo = ejemplo_reducido

        secuencia_lista[i] = ejemplo_reducido

    print("Longitud promedio de los ejemplos: {}".format(suma/len(secuencia_lista)))
    return secuencia_lista

def otro_cargar_ejemplos(etiquetas_neural_networks_ruta,
                         interacciones_lista_ruta,
                         excluir_interacciones_lista,
                         ejemplos_cantidad,
                         porcentaje_prueba,
                         publicaciones_directorio,
                         maxima_longitud_ejemplos):
                        #  out_interacciones_ruta,
                        #  incluir_sin_interacciones=True,
                        #  top_palabras=150,
                        #  max_longitud=500,
                        #  embeddings_file="glove.6B.50d.txt",
                        #  sin_interaccion_a_incluir=2944,
                        #  randomize=True,
                        #  vocabulario_global=True):

    # # Se cargan las publicaciones en un diccionario: publicaciones_dict[pmid] = contenido
    # print("Cargando diccionario de publicaciones.")
    # publicaciones_dict = dict()
    # publicaciones_en_directorio = os.listdir(publicaciones_directorio)
    # for archivo in publicaciones_en_directorio:
    #     pmid = archivo.split(".")[0]
    #     archivo_ruta = os.path.join(publicaciones_directorio, archivo)
    #     with open(archivo_ruta, encoding="utf8") as publicacion:
    #         texto = publicacion.read()
    #         publicaciones_dict[pmid] = texto
    # print("Diccionario de publicaciones cargado.")

    # # Se cargan las interacciones fármaco-gen en una lista
    # print("Cargando las interacciones fármaco-gen en una lista.")
    # ifg_lista = list()
    # with open(etiquetas_neural_networks_ruta, encoding="utf8") as etiquetas_neural_networks:
    #     lector_csv = csv.reader(etiquetas_neural_networks, delimiter=',', quoting=csv.QUOTE_ALL)
    #     for ifg in lector_csv:
    #         if ifg[3] not in excluir_interacciones_lista:
    #             ifg_lista.append(ifg)
    # print("Lista de interacciones fármaco-gen cargadas.")
 
    # # Se generan las listas de ejemplos
    # print("Generando listas de ejemplos.")
    # ejemplos_lista = list() # Esta lista se utiliza para crear el vocabulario
    # for i in range(0, len(ifg_lista), 1):
    #     ejemplos_lista.append(publicaciones_dict[ifg_lista[i][0]])
    # print("Listas de ejemplos generadas.")

    # Se genera el vocabulario
    print("Generando vocabulario.")
    with open("vocabulario.pickle", "rb") as handle: # Cargar vocabulario desde el disco
        vocabulario = pickle.load(handle)
    # vocabulario = text.Tokenizer()
    # vocabulario.fit_on_texts(ejemplos_lista)
    # with open("vocabulario.pickle", "wb") as handle: # Guardar vocabulario en disco
    #     pickle.dump(vocabulario, handle, protocol=pickle.HIGHEST_PROTOCOL)
    top_palabras_frecuentes = len(vocabulario.word_index)
    # print("Cantidad de palabras en el vocabulario: {}".format(top_palabras_frecuentes))
    print("Vocabulario generado.")

    # Se convierten los ejemplos en secuencias de números
    print("Generando secuencias.")
    with open("secuencias_dict.pickle", "rb") as handle: # Cargar vocabulario desde el disco
        secuencias_dict = pickle.load(handle)
    # publicaciones_lista = list()
    # publicaciones_pmids = list()
    # for pmid, contenido in publicaciones_dict.items():
    #     publicaciones_pmids.append(pmid)
    #     publicaciones_lista.append(contenido)
    # secuencias_lista = vocabulario.texts_to_sequences(publicaciones_lista)
    # secuencias_dict = dict()
    # for i in range(0, len(publicaciones_pmids), 1):
    #     secuencias_dict[publicaciones_pmids[i]] = secuencias_lista[i]
    # with open("secuencias_dict.pickle", "wb") as handle: # Guardar vocabulario en disco
        # pickle.dump(secuencias_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Secuencias generadas.")

    # One hot encoding de la salida
    interacciones_lista = list()
    interacciones_numeros_lista = list()
    with open(interacciones_lista_ruta) as interacciones:
        contador = 0
        for interaccion in interacciones:
            interacciones_lista.append(interaccion.strip())
            interacciones_numeros_lista.append(contador)
            contador += 1
    interacciones_numeros_lista = np_utils.to_categorical(interacciones_numeros_lista)
    interacciones_numeros_dict = dict()
    for i in range(0, len(interacciones_lista), 1):
        interacciones_numeros_dict[interacciones_lista[i]] = interacciones_numeros_lista[i]

    # Se cargan en dos lista las interacciones fármaco-gen para entrenamiento y prueba
    print("Cargando listas de interacciones fármaco-gen para entrenamiento y prueba.")
    ifg_balanceadas_entrenamiento_lista, ifg_balanceadas_prueba_lista = ff.balancear_clases(etiquetas_neural_networks_ruta, # Archivo de etiquetas: pmid, gen, droga, interacción
                                                                                            interacciones_lista_ruta, # Lista de etiquetas a considerar
                                                                                            excluir_interacciones_lista, # Lista de interacciones que no se cargarán
                                                                                            ejemplos_cantidad, # Cantidad de ejemplos a cargar
                                                                                            porcentaje_prueba) # Porcentaje de los ejemplos que se utilizarán para la prueba
    print("Listas de interacciones fármaco-gen para entrenamiento y prueba cargadas.")
    
    # Se generan las listas de ejemplos
    print("Generando listas de ejemplos.")
    ejemplos_x_entrenamiento_lista = list()
    ejemplos_y_entrenamiento_lista = list()
    ejemplos_x_prueba_lista = list()
    ejemplos_y_prueba_lista = list()
    for i in range(0, len(ifg_balanceadas_entrenamiento_lista), 1):
        ejemplos_x_entrenamiento_lista.append(secuencias_dict[ifg_balanceadas_entrenamiento_lista[i][0]])
        ejemplos_y_entrenamiento_lista.append(interacciones_numeros_dict[ifg_balanceadas_entrenamiento_lista[i][3]])
    for i in range(0, len(ifg_balanceadas_prueba_lista), 1):
        ejemplos_x_prueba_lista.append(secuencias_dict[ifg_balanceadas_prueba_lista[i][0]])
        ejemplos_y_prueba_lista.append(interacciones_numeros_dict[ifg_balanceadas_prueba_lista[i][3]])
    print("Listas de ejemplos generadas.")

    # # Se convierten los ejemplos en secuencias de números
    # print("Generando secuencias.")
    # ejemplos_x_entrenamiento_secuencia_lista = vocabulario.texts_to_sequences(ejemplos_x_entrenamiento_lista)
    # ejemplos_x_prueba_secuencia_lista = vocabulario.texts_to_sequences(ejemplos_x_prueba_lista)
    # print("Secuencias generadas.")

    # Se ajustan las secuencias a la longitud deseada
    print("Ajustando secuencias a la longitud deseada.")
    ejemplos_x_entrenamiento_secuencia_lista = acortar_secuencia(ejemplos_x_entrenamiento_lista, vocabulario, ifg_balanceadas_entrenamiento_lista, maxima_longitud_ejemplos)
    ejemplos_x_prueba_secuencia_lista = acortar_secuencia(ejemplos_x_prueba_lista, vocabulario, ifg_balanceadas_prueba_lista, maxima_longitud_ejemplos)

    ejemplos_x_entrenamiento_secuencia_ajustada_lista = sequence.pad_sequences(sequences=ejemplos_x_entrenamiento_secuencia_lista,
                                                                               maxlen=maxima_longitud_ejemplos,
                                                                               padding="post",
                                                                               truncating="post")
    ejemplos_x_prueba_secuencia_ajustada_lista = sequence.pad_sequences(sequences=ejemplos_x_prueba_secuencia_lista,
                                                                        maxlen=maxima_longitud_ejemplos,
                                                                        padding="post",
                                                                        truncating="post")
    print("Ajuste de longitud de secuencias terminado.")

    return (ejemplos_x_entrenamiento_secuencia_ajustada_lista, np.asarray(ejemplos_y_entrenamiento_lista)), (ejemplos_x_prueba_secuencia_ajustada_lista, np.asarray(ejemplos_y_prueba_lista)), top_palabras_frecuentes, vocabulario

# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    etiquetas_neural_networks_ruta = "etiquetas_neural_networks.csv"
    interacciones_lista_ruta = "interacciones_lista.txt"
    excluir_interacciones_lista = []
    ejemplos_cantidad = 50
    porcentaje_prueba = 0.2
    publicaciones_directorio = "replaced_new"
    maxima_longitud_ejemplos = 100
    # etiquetas_neural_networks_ruta = "test_etiquetas"
    # ejemplos_directorio = "."
    # embeddings_ruta = "E:/Descargas/Python/glove.6B.300d.txt"

    # (xe, ye), (xt, yt) = cargar_ejemplos(etiquetas_neural_networks_ruta, ejemplos_directorio,
    #                                  out_interacciones_ruta, porcentaje_test=0.2,
    #                                  sin_interaccion_a_incluir=1)
    # print("Cantidad x entrenamiento:", len(xe))
    # print("Cantidad y entrenamiento:", len(ye))
    # print("Cantidad x test:", len(xt))
    # print("Cantidad y test:", len(yt))

    (x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = otro_cargar_ejemplos(etiquetas_neural_networks_ruta,
                                                                                    interacciones_lista_ruta,
                                                                                    excluir_interacciones_lista,
                                                                                    ejemplos_cantidad,
                                                                                    porcentaje_prueba,
                                                                                    publicaciones_directorio,
                                                                                    maxima_longitud_ejemplos)

    # print(x_entrenamiento[0])                                                                    
    # print(y_entrenamiento[0])
    # print(x_prueba[15])
    # print(y_prueba[15])