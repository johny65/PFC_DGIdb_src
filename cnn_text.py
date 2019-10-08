# Compatibilidad entre Python 2 y 3
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd # Para el manejo de datos
import numpy as np # Para operaciones matemáticas de nivel inicial

# Preprocesamiento de los datos (texto)
from keras.preprocessing import text,sequence 

# Modelo
from keras.models import Sequential
from keras.layers import Embedding,Conv1D,MaxPooling1D,Flatten,Dropout,Dense # SeparableConv1D, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import funciones_auxiliares as fa
import cargar_imdb as ci

import explore_data as ed

# Carga de datos
(x_entrenamiento,y_entrenamiento),(x_prueba,y_prueba) = ci.load_imdb_sentiment_analysis_dataset('D:\\Descargas\\Python\\neural_networks')

# num_classes = ed.get_num_classes(y_entrenamiento)
# print(num_classes)

# print(ed.get_num_words_per_sample(x_entrenamiento))
# ed.plot_sample_length_distribution(x_entrenamiento)

# Variables
top_palabras_frecuentes = 20000 # Cantidad de palabras en el vocabulario
maxima_longitud_ejemplos = 500 # Máxima longitud de los ejemplos de entrada
dimension_vectores_embedding = 300 # Cantidad de elementos de los vectores de embedding
embedding_entrenable = True # Bandera de adaptación de pesos para embeddings preentrenados
porcentaje_dropeo = 0.5 # Pone en 0 el #% de los datos aleatoriamente
cantidad_filtros = 16 # Cantidad de filtro de convolución
dimension_kernel = 3 # De 1x3
dimension_pooling = 2
neuronas_ocultas = 32
velocidad_aprendizaje = 1e-3
cantidad_epocas = 100
porcentaje_validacion = 0.2 # 20% de ejemplos usados para validar. Son tomados desde el final
dimension_bacha = 512

# Crea el vocabulario a partir de los datos de entrenamiento
tokenizer = text.Tokenizer(num_words=top_palabras_frecuentes)
tokenizer.fit_on_texts(x_entrenamiento)

# Vectorización de los datos de entrada (entrenamiento y prueba): "Hola querido mundo" -> [3857 274 982]
x_entrenamiento_vectorizado = tokenizer.texts_to_sequences(x_entrenamiento)
x_prueba_vectorizado = tokenizer.texts_to_sequences(x_prueba)

# Obtener longitud del ejemplo más largo
maxima_longitud = len(max(x_entrenamiento_vectorizado,key=len))
if maxima_longitud > maxima_longitud_ejemplos:
    maxima_longitud = maxima_longitud_ejemplos

# Se arreglan los ejemplos para que todos tengan la misma longitud
x_entrenamiento_arreglado = sequence.pad_sequences(x_entrenamiento_vectorizado,maxlen=maxima_longitud)
x_prueba_arreglado = sequence.pad_sequences(x_prueba_vectorizado,maxlen=maxima_longitud)

indice_palabras = tokenizer.word_index

# Se cargan vectores de embedding preentrenados (GloVe)
embeddings_index = dict()
f = open('glove.6B.300d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((top_palabras_frecuentes,dimension_vectores_embedding))
for word, index in tokenizer.word_index.items():
    if index > top_palabras_frecuentes - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

# Modelo secuencial (una capa detrás de la otra): Perceptrón multicapa (Arquitectura)
modelo = Sequential()

'''Dropout: se utiliza para evitar el sobreentrenamiento'''

# model.add(Embedding(input_dim=top_palabras_frecuentes,
#                     output_dim=dimension_vectores_embedding,
#                     input_length=maxima_longitud))

modelo.add(Embedding(input_dim=top_palabras_frecuentes,
                    output_dim=dimension_vectores_embedding,
                    input_length=maxima_longitud,
                    weights=[embedding_matrix],
                    trainable=embedding_entrenable))

# modelo.add(Dropout(porcentaje_dropeo))

modelo.add(Conv1D(filters=cantidad_filtros,
                 kernel_size=dimension_kernel,
                 activation='relu',
                 bias_initializer='random_uniform',
                 padding='same'))
modelo.add(MaxPooling1D(pool_size=dimension_pooling))

modelo.add(Conv1D(filters=cantidad_filtros,
                 kernel_size=dimension_kernel,
                 activation='relu',
                 bias_initializer='random_uniform',
                 padding='same'))
modelo.add(MaxPooling1D(pool_size=dimension_pooling))

modelo.add(Flatten()) # input_shape=(imagen_ancho,imagen_alto)

# Capa de entrada
modelo.add(Dropout(porcentaje_dropeo))
modelo.add(Dense(neuronas_ocultas,activation='relu'))

# Capa de salida
modelo.add(Dropout(porcentaje_dropeo))
modelo.add(Dense(1,activation='sigmoid'))

'''
Activations: Funciones de activación
    elu
    softmax
    selu
    softplus
    softsign
    relu
    tanh
    sigmoid
    hard_sigmoid
    exponential
    linear
'''

# Proceso de aprendizaje
optimizador = Adam(lr=velocidad_aprendizaje)
modelo.compile(optimizer=optimizador, # Velocidad de aprendizaje
               loss='binary_crossentropy', # Función de error
               metrics=['accuracy']) # Análisis del modelo (Tasa de acierto))

'''
Optimizers (Velocidad de aprendizaje):
    sgd
    rmsprop
    adagrad
    adadelta
    adam
    adamax
    nadam
    
Loss: Función que compara la salida deseada con la calculada. Criterio de error.
Se busca minimizar esta función mediante el método del gradiente descendente.
    mean_squared_error
    mean_absolute_error
    mean_absolute_percentage_error
    mean_squared_logarithmic_error
    squared_hinge
    hinge
    categorical_hinge
    logcosh
    huber_loss
    categorical_crossentropy
    sparse_categorical_crossentropy
    binary_crossentropy
    kullback_leibler_divergence
    poisson
    cosine_proximity
    is_categorical_crossentropy
    
Metrics: Análisis de desempeño del modelo
    accuracy
    binary_accuracy
    categorical_accuracy
    sparse_categorical_accuracy
    top_k_categorical_accuracy
    sparse_top_k_categorical_accuracy
    cosine_proximity
    clone_metric
    clone_metrics
'''

# Create callback for early stopping on validation loss. If the loss does
# not decrease in two consecutive tries, stop training.
callbacks = [EarlyStopping(monitor='val_accuracy',patience=1)] # 'val_loss'

# Detalles del modelo neuronal
modelo.summary()

# Entrenamiento del modelo
registro = modelo.fit(x_entrenamiento_arreglado,
                      y_entrenamiento,
                      epochs=cantidad_epocas,
                      callbacks=callbacks,
                      validation_data=(x_prueba_arreglado,y_prueba), # porcentaje_validacion
                      verbose=1,
                      batch_size=dimension_bacha)

fa.graficas(registro)

# Evaluación del modelo
# prueba_error,prueba_acierto = model.evaluate(x_prueba,y_prueba)

# Hacer predicciones
# predicciones = model.predict(x_prueba)