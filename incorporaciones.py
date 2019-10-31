import sys


def interacciones_tipos(datos_entrenamiento):
    interacciones = set()
    for linea in datos_entrenamiento:
        datos = linea.split("\t")
        # pmid = datos[0]
        # gen = datos[1]
        # droga = datos[2]
        interaccion = datos[3]
        # abstract = datos[4:]
        interacciones.add(interaccion)
    # interacciones_mapa = list(interacciones).sort()
    interacciones_mapa = list(interacciones)
    interacciones_mapa.sort()
    return interacciones_mapa

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

    datos_entrenamiento = open("D:\Descargas\Python\PFC_DGIdb_src\datos_entrenamiento.tsv",encoding="utf8")

    interacciones_mapa = interacciones_tipos(datos_entrenamiento)

    print(interacciones_mapa)

    datos_entrenamiento.close()