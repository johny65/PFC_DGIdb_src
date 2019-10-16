# Compatibilidad entre Python 2 y 3
from __future__ import absolute_import, division, print_function, unicode_literals

# import pandas as pd # Para el manejo de datos
import numpy as np # Para operaciones matemáticas de nivel inicial

# Preprocesamiento de los datos (texto)
from keras.preprocessing import text,sequence 

# Modelo
# from keras.models import Sequential
# from keras.layers import Embedding,Conv1D,MaxPooling1D,Flatten,Dropout,Dense # SeparableConv1D, GlobalAveragePooling1D
# from keras.callbacks import EarlyStopping
# from keras.optimizers import Adam

import os

# import funciones_auxiliares as fa
# import cargar_imdb as ci

# import explore_data as ed

# Se cargan vectores de embedding preentrenados (GloVe)
# embeddings_index = dict()
# f = open('glove.pdf.txt',encoding="utf8")
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='str')
#     embeddings_index[word] = coefs
# f.close()

datos = []
archivo = open('glove.pdf.txt',encoding="utf8")
                    # train_texts.append(f.read())
datos.append(archivo.read())

print(datos)

# datos = np.loadtxt('glove.pdf.txt',dtype=str)

# # print(embeddings_index)
# print(datos)

# # Crea el vocabulario a partir de los datos de entrenamiento
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(datos)

# # Vectorización de los datos de entrada (entrenamiento y prueba): "Hola querido mundo" -> [3857 274 982]
datos_vectorizado = tokenizer.texts_to_sequences(datos)

print(datos_vectorizado[0])