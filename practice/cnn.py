# Compatibilidad entre Python 2 y 3
from __future__ import absolute_import, division, print_function, unicode_literals

# Importación de datos
from keras.datasets import mnist,fashion_mnist
# Simplificación de la escritura del modelo
from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout,Conv2D,MaxPooling2D
from keras.utils import np_utils

# Artilugios matemáticos
# import numpy as np

import funciones_auxiliares as fa

# Cargar datos
# (entradas_entrenamiento,salidas_entrenamiento), (entrada_pruebas,salidas_prueba)
# (x_entrenamiento,y_entrenamiento), (x_prueba,y_prueba) = mnist.load_data()
(x_entrenamiento,y_entrenamiento), (x_prueba,y_prueba) = fashion_mnist.load_data()

'''
mnist contiene imágenes de números escritos a mano
fashion_mnist contiene imágenes de ropa

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

cantidad_ejemplos_e = x_entrenamiento.shape[0] # 60.000
cantidad_ejemplos_p = x_prueba.shape[0] # 10.000
imagen_ancho = x_entrenamiento.shape[1] # 28
imagen_alto = x_entrenamiento.shape[2] # 28

'''
El método Conv2D de Keras necesita como entrada un objeto de 3 dimensiones.
Ancho, alto y canales de color. Se añade una dimensión más a los datos.
'''
x_entrenamiento = x_entrenamiento.reshape(cantidad_ejemplos_e,imagen_ancho,imagen_alto,1)
x_prueba = x_prueba.reshape(cantidad_ejemplos_p,imagen_ancho,imagen_alto,1)
canales = x_entrenamiento.shape[3]

# One hot encoding
y_entrenamiento = np_utils.to_categorical(y_entrenamiento)
y_prueba = np_utils.to_categorical(y_prueba)

clases = y_entrenamiento.shape[1] # 10 clases
neuronas_entrada = 100

# Modelo: Perceptrón multicapa (Arquitectura)
model = Sequential()

cant_conv = 32
kernel = (3,3)
formato1 = (imagen_ancho,imagen_alto,canales)

# 3*3*32 + 32 = 320 parámetros

model.add(Conv2D(cant_conv,kernel,input_shape=formato1,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

formato2 = ((imagen_ancho-2)/2,(imagen_alto-2)/2,canales)

model.add(Conv2D(cant_conv,kernel,input_shape=formato2,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

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

# Detalles del modelo neuronal
model.summary()

epocas = 10
datos_val = 0.1 # Porcentaje de ejemplos usados para validar. Son tomados desde el final

# Entrenamiento del modelo
registro = model.fit(x_entrenamiento,y_entrenamiento,epochs=epocas,verbose=1,validation_split=datos_val)

# fa.graficas(registro)

# Evaluación del modelo
# prueba_error,prueba_acierto = model.evaluate(x_prueba,y_prueba)

# Hacer predicciones
# predicciones = model.predict(x_prueba)


















