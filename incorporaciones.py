import sys
import csv
import numpy as np
import re

# Preprocesamiento de los datos (texto)
from keras.preprocessing import text,sequence 

def cargar_datos_entrenamiento(datos_entrenamiento_ruta,alias_gen_entrenamiento,alias_droga_entrenamiento,embeddings_ruta):
    '''

    '''

    embeddings_dict = dict()
    with open(embeddings_ruta,encoding="utf8") as embeddings:
        for fila in embeddings:
            datos = fila.split()
            palabra = datos[0]
            embedding = np.asarray(datos[1:],dtype='float32')
            embeddings_dict[palabra] = embedding

    print("Embeddings cargados")

    # alias_gen_conjunto = set()
    # with open(alias_gen_entrenamiento,encoding="utf8") as alias_gen_tsv:
    #     lector_tsv = csv.reader(alias_gen_tsv, delimiter='\t', quoting=csv.QUOTE_ALL)
    #     for fila in lector_tsv:
    #         for e in fila:
    #             alias_gen_conjunto.add(e.lower())        

    # print("Aliases de genes cargados")

    # alias_droga_conjunto = set()
    # with open(alias_droga_entrenamiento,encoding="utf8") as alias_droga_tsv:
    #     lector_tsv = csv.reader(alias_droga_tsv, delimiter='\t', quoting=csv.QUOTE_ALL)
    #     for fila in lector_tsv:
    #         for e in fila:
    #             alias_droga_conjunto.add(e.lower())

    # print("Aliases de drogas cargados")

    abstracts_dict = dict() # Mapa / Diccionario
    # abstracts_list = list() # list() == []
    with open(datos_entrenamiento_ruta,encoding="utf8") as archivo_tsv:
        lector_tsv = csv.reader(archivo_tsv, delimiter='\t', quoting=csv.QUOTE_ALL)
        for fila in lector_tsv:
            # 0: pmid, 1: gen, 2: droga, 3: interacción, 4: abstract
            s = fila[4].lower()
            s = re.sub('[^a-zA-Z ]','',s)
            abstracts_dict[fila[0]] = s # abstracts_dict['pmid'] = abstract.str
            # abstracts_list.append(fila[4])

    # print(abstracts_dict['17288876'])
    # print(abstracts_dict)
    # print(abstracts_list[4])

    print("Abstracts cargados")

    sin_embedding = open("sin_embedding6.tsv","w",encoding="utf8")
    escritor_tsv = csv.writer(sin_embedding,delimiter='\t',lineterminator="\n",quoting=csv.QUOTE_ALL)

    for fila in abstracts_dict:
        palabras = abstracts_dict[fila].split()
        lista = list()
        for palabra in palabras:
            if palabra not in embeddings_dict:
                lista.append(palabra)
        escritor_tsv.writerow(lista)

    sin_embedding.close()

    print("Archivo sin_embedding creado")

    # lista_gen = list()
    # lista_droga = list()
    # for fila in abstracts_dict:
    #     palabras = abstracts_dict[fila].split()
        
        # g = []
        # for gen in alias_gen_conjunto:
        #     if gen in palabras:
        #         g.append(gen)
        # lista_gen.append(g)

        # d = []
        # for droga in alias_droga_conjunto:
        #     if droga in palabras:
        #         d.append(droga)
        # lista_droga.append(d)

    # print(lista_gen)
    # print(lista_droga)

    # Crea el vocabulario a partir de los datos de entrenamiento
    # tokenizer = text.Tokenizer()
    # tokenizer.fit_on_texts(abstracts_list)

    # print(len(tokenizer.word_index))
    # print(tokenizer.document_count)

    # # Vectorización de los datos de entrada (entrenamiento y prueba): "Hola querido mundo" -> [3857 274 982]
    # x_entrenamiento_vectorizado = tokenizer.texts_to_sequences(x_entrenamiento)
    # x_prueba_vectorizado = tokenizer.texts_to_sequences(x_prueba)

    # # Obtener longitud del ejemplo más largo
    # maxima_longitud = len(max(x_entrenamiento_vectorizado,key=len))
    # if maxima_longitud > MAXIMA_LONGITUD_EJEMPLOS:
    #     maxima_longitud = MAXIMA_LONGITUD_EJEMPLOS

    # # Se arreglan los ejemplos para que todos tengan la misma longitud
    # x_entrenamiento_arreglado = sequence.pad_sequences(x_entrenamiento_vectorizado,maxlen=maxima_longitud)
    # x_prueba_arreglado = sequence.pad_sequences(x_prueba_vectorizado,maxlen=maxima_longitud)

    return

# def interacciones_tipos(datos_entrenamiento):
#     interacciones = set()
#     for linea in datos_entrenamiento:
#         datos = linea.split("\t")
#         # pmid = datos[0]
#         # gen = datos[1]
#         # droga = datos[2]
#         interaccion = datos[3]
#         # abstract = datos[4:]
#         interacciones.add(interaccion)
#     # interacciones_mapa = list(interacciones).sort()
#     interacciones_mapa = list(interacciones)
#     interacciones_mapa.sort()
#     return interacciones_mapa

# def incorporacion_glove():
#     '''
#     Carga los vectores de embedding preentrenados de GloVe
#     '''
    
#     embeddings_index = dict()
#     f = open('glove.6B.300d.txt',encoding="utf8")
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#     f.close()

#     embedding_matrix = np.zeros((TOP_PALABRAS_FRECUENTES,DIMENSION_VECTORES_EMBEDDING))
#     for word, index in tokenizer.word_index.items():
#         if index > TOP_PALABRAS_FRECUENTES - 1:
#             break
#         else:   
#             embedding_vector = embeddings_index.get(word)
#             if embedding_vector is not None:
#                 embedding_matrix[index] = embedding_vector  


# def sample_to_number():

#     return 

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Forma de uso: {} entrada salida".format(sys.argv[0]))
        exit()
    
    datos_entrenamiento_ruta = "E:/Descargas/Python/PFC_DGIdb_src/scraping/varios/datos_entrenamiento.tsv"
    alias_gen_entrenamiento = "E:/Descargas/Python/PFC_DGIdb_src/gen_alias_entrenamiento.tsv"
    alias_droga_entrenamiento = "E:/Descargas/Python/PFC_DGIdb_src/droga_alias_entrenamiento.tsv"
    embeddings_ruta = "E:/Descargas/Python/glove.6B.300d.txt"

    cargar_datos_entrenamiento(datos_entrenamiento_ruta,alias_gen_entrenamiento,alias_droga_entrenamiento,embeddings_ruta)