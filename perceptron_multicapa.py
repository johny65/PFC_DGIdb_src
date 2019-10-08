# Compatibilidad entre Python 2 y 3
from __future__ import absolute_import, division, print_function, unicode_literals

# Importación de datos
from keras.datasets import mnist
# Simplificación de la escritura del modelo
from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout
from keras.utils import np_utils

# Artilugios matemáticos
# import numpy as np

import funciones_auxiliares as fa

# Cargar datos
# (entradas_entrenamiento,salidas_entrenamiento), (entrada_pruebas,salidas_prueba)
(x_entrenamiento,y_entrenamiento), (x_prueba,y_prueba) = mnist.load_data()

'''
mnist contiene imágenes de números escritos a mano

x_entrenamiento: 60.000 entradas de 28x28
y_entrenamiento: 60.000 etiquetas del 0 al 9
x_prueba: 10.000 entradas de 28x28
y_prueba: 10.000 etiquetas del 0 al 9
'''

# Normalización de los datos
'''
Las redes neuronales son dependientes de la magnitud de los datos. Es
recomendable normalizar los datos siempre.
'''
x_entrenamiento = x_entrenamiento/255.0
x_prueba = x_prueba/255.0

# fa.mostrar_imagen(x_entrenamiento[0],y_entrenamiento[0])
# fa.mostrar_cuadricula(x_entrenamiento,y_entrenamiento)

# Coloca True donde la etiqueta es 5 y False en los demás casos
# es_cinco_entrenamiento = y_entrenamiento == 5
# es_cinco_prueba = y_prueba == 5

# cantidad_ejemplos = x_entrenamiento.shape[0] # 60.000
imagen_ancho = x_entrenamiento.shape[1] # 28
imagen_alto = x_entrenamiento.shape[2] # 28

# One hot encoding
y_entrenamiento = np_utils.to_categorical(y_entrenamiento)
y_prueba = np_utils.to_categorical(y_prueba)

clases = y_entrenamiento.shape[1] # 10 clases
neuronas_entrada = 100

# Modelo: Perceptrón multicapa (Arquitectura)
model = Sequential()
# 60000 entradas de 1x784 (Resultado de aplanar 28x28)
model.add(Flatten()) # input_shape=(imagen_ancho,imagen_alto)

# Capa de entrada
model.add(Dropout(0.4)) # Pone en 0 el 40% de los datos aleatoriamente
model.add(Dense(neuronas_entrada,activation='relu'))

# Capa de salida
model.add(Dropout(0.4)) # Pone en 0 el 40% de los datos aleatoriamente
model.add(Dense(clases,activation='softmax')) 

'''
Dropout: sirve para evitar el sobreentrenamiento

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
model.compile(optimizer='adam', # Velocidad de aprendizaje
              loss='categorical_crossentropy', # Función de error
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

epocas = 10
datos_val = 0.1 # Porcentaje de ejemplos usados para validar. Son tomados desde el final

# Entrenamiento del modelo
registro = model.fit(x_entrenamiento,y_entrenamiento,epochs=epocas,verbose=1,validation_split=datos_val)

# Detalles del modelo neuronal
model.summary()

fa.graficas(registro)

# Evaluación del modelo
# prueba_error,prueba_acierto = model.evaluate(x_prueba,es_cinco_prueba)

# Hacer predicciones
# predicciones = model.predict(x_prueba)


















