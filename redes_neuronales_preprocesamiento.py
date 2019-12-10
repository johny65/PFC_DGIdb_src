import os
import numpy as np
from keras.preprocessing import text,sequence
import datos_preprocesamiento
import csv

def cargar_ejemplos(ejemplos_directorio): # etiquetas_neural_networks_ruta,
    '''
    Carga los ejemplos para las redes en una lista de cadenas
    '''
    pmids = list()
    genes = list()
    drogas = list()
    interacciones = list()
    with open(etiquetas_neural_networks_ruta,encoding="utf8") as enn_csv:
        lector_csv = csv.reader(enn_csv,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            pmids.append(fila[0])
            genes.append(fila[1])
            drogas.append(fila[2])
            interacciones.append(fila[3])

    ejemplos_lista = list()
    for i in range(0,len(pmids),1):
        archivo_nombre = pmids[i] + ".txt"
        archivo_ruta = os.path.join(ejemplos_directorio,archivo_nombre)
        with open(archivo_ruta,encoding="utf8") as ejemplo:
            ejemplos_lista.append(ejemplo.read())
            print("Ejemplo {} con etiquetas {},{},{} cargado.".format(pmids[i],genes[i],drogas[i],interacciones[i]))
    return ejemplos_lista,genes,drogas

def arreglar_ejemplos(ejemplos_lista):
    '''
    Convierte los ejemplos (strings) a listas de números de un vocabulario y completa con ceros para que todos tengan la misma longitud (longitud del ejemplo más largo).
    '''
    vocabulario = text.Tokenizer() # Crea un vocabulario
    vocabulario.fit_on_texts(ejemplos_lista) # Llena el vocabulario en base al contenido de las publicaciones
    ejemplos_lista_vectorizados = vocabulario.texts_to_sequences(ejemplos_lista) # "Hola querido mundo" -> [3857 274 982]
    # maxima_longitud = len(max(ejemplos_lista_vectorizados,key=len)) # Longitud del ejemplo más largo
    ejemplos_lista_arreglados = sequence.pad_sequences(ejemplos_lista_vectorizados) # ,maxlen=maxima_longitud
    vocabulario.num_words
    return ejemplos_lista_arreglados,vocabulario

def generar_entradas_salidas_redes(ejemplos_lista_arreglados,vocabulario,embeddings_dict,genes,drogas):
    embeddings = list()
    cantidad_palabras = vocabulario.num_words
    dimension_embedding = 302
    gen = np.zeros((1,dimension_embedding))
    gen[0][-2] = 1
    droga = np.zeros((1,dimension_embedding))
    droga[0][-1] = 1
    for i in range(0,len(ejemplos_lista_arreglados),1):
        arreglo_base = np.zeros((cantidad_palabras,dimension_embedding))
        ejemplo_arreglado = ejemplos_lista_arreglados[i]
        for j in range(0,len(ejemplo_arreglado),1):
            palabra_numero = ejemplo_arreglado[j]
            palabra_texto = vocabulario.get(palabra_numero)
            if palabra_texto == "{[" + genes[i] + "]}":
                arreglo_base[j] = gen
            elif palabra_texto == "{[" + drogas[i] + "]}":
                arreglo_base[j] = droga
            else:
                embedding_vector = embeddings_dict.get(palabra_texto)
                if embedding_vector is not None:
                    arreglo_base[j] = embedding_vector
        embeddings.append(arreglo_base)
    return embeddings
    

if __name__ == "__main__":

    ejemplos_directorio = "E:/Descargas/Python/PFC_DGIdb_src/scraping/files/labeled/txt/txt_ungreek"
    embeddings_ruta = "E:/Descargas/Python/glove.6B.300d.txt"

    

    # ejemplos_lista = cargar_ejemplos(ejemplos_directorio)
    # print("Ejemplos cargados.")
    # print(len(ejemplos_lista))
    # ejemplos_lista_arreglados,vocabulario = arreglar_ejemplos(ejemplos_lista)
    # print("Maxima longitud: {}".format(maxima_longitud))
    # print(ejemplos_lista_arreglados[0])
    # print(len(ejemplos_lista_arreglados[0]))
    # print(vocabulario.word_index["hello"])

    embeddings_dict = datos_preprocesamiento.cargar_embeddings(embeddings_ruta)
    
    # print(len(embeddings_dict["hello"]))

    # a = [[1],[1,2],[1,2,3],[1],[1,2,3,4],[2,3,5],[1,2],[3,5]]
    # print(len(max(a,key=len)))

    # gen = np.zeros((1,10))
    # gen[0][-2] = 1
    # droga = np.zeros((1,10))
    # droga[0][-1] = 1
    # print(gen)
    # print(droga)