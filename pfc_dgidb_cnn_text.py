# Paquetes
from __future__ import absolute_import, division, print_function, unicode_literals # Compatibilidad entre Python 2 y 3
from redes_neuronales_preprocesamiento import cargar_ejemplos # Carga de ejemplos
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Concatenate, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam,SGD # SGD: Gradiente descendiente
import funciones_auxiliares as fa
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Para utilizar CPU en lugar de GPU

''' Carga de datos '''
etiquetas_neural_networks_ruta = "etiquetas_neural_networks2.csv"
ejemplos_directorio = "replaced"
out_interacciones_ruta = "interacciones_lista.txt"
embeddings_file = "glove.6B.50d.txt"
# embeddings_file = "glove.6B.100d.txt"
# embeddings_file = "glove.6B.200d.txt"
# embeddings_file = "glove.6B.300d.txt"
top_palabras_frecuentes = 100
maxima_longitud_ejemplos = 500

# maxima_longitud_ejemplos: 300 -> top_palabras_frecuentes: 64
# maxima_longitud_ejemplos: 500 -> top_palabras_frecuentes: 103
# maxima_longitud_ejemplos: 1000 -> top_palabras_frecuentes: 552
# maxima_longitud_ejemplos: 1500 -> top_palabras_frecuentes: 790
# maxima_longitud_ejemplos: 2000 -> top_palabras_frecuentes: 1112
# maxima_longitud_ejemplos: 2500 -> top_palabras_frecuentes: 1298
# maxima_longitud_ejemplos: 3000 -> top_palabras_frecuentes: 1421

x_entrenamiento, y_entrenamiento = cargar_ejemplos(etiquetas_neural_networks_ruta,
                                                    ejemplos_directorio,
                                                    out_interacciones_ruta,
                                                    embeddings_file=embeddings_file,
                                                    top_palabras=top_palabras_frecuentes,
                                                    max_longitud=maxima_longitud_ejemplos,
                                                    incluir_sin_interacciones = True,
                                                    sin_interaccion_a_incluir = 3144) # 1131

cantidad_ejemplos = x_entrenamiento.shape[0]
dimension_embedding = x_entrenamiento.shape[2] # Columnas
cantidad_clases = y_entrenamiento.shape[1]

# Formato de entrada para la red
# x_entrenamiento = x_entrenamiento.reshape(cantidad_ejemplos, dimension_embedding, maxima_longitud_ejemplos, canales)

''' Parámetros del modelo '''
k = 5 # Número de particiones para la validación cruzada
PORCENTAJE_DROPEO = 0.3 # Pone en 0 el #% de los datos aleatoriamente
CANTIDAD_FILTROS = 100 # Cantidad de filtro de convolución
DIMENSION_KERNEL = (3, 4, 5)
NEURONAS_OCULTAS = 512
NEURONAS_SALIDA = cantidad_clases
CANTIDAD_EPOCAS = 100
PORCENTAJE_VALIDACION = 0.2
DIMENSION_BACHA = 32
ACTIVACION_OCULTA = 'relu'
ACTIVACION_SALIDA = 'softmax'
OPTIMIZADOR = 'adam'
# VELOCIDAD_APRENDIZAJE = 1e-7
FUNCION_ERROR = 'categorical_crossentropy'
''' --------------------- '''

folds_dict = fa.kfolding(k, cantidad_ejemplos, PORCENTAJE_VALIDACION) # Se obtienen las particiones para realizar la validación cruzada

suma_acierto_entrenamiento = 0
suma_acierto_validacion = 0
# suma_acierto_prueba = 0

resultados_finales = list()

for i in range(0,k,1):
    modelo_cnn = Sequential()
    formato_entrada = (maxima_longitud_ejemplos, dimension_embedding)

    '''Intento de entrenar con kernels de distintos tamaños al mismo tiempo'''
    submodelos = list()
    for i in range(0, len(DIMENSION_KERNEL), 1):
        submodelos[i] = Sequential()
        submodelos[i].add(Conv1D(input_shape=formato_entrada,
                          filters=CANTIDAD_FILTROS,
                          kernel_size=DIMENSION_KERNEL[i],
                          padding='same',
                          bias_initializer='random_uniform'))
        submodelos[i].add(MaxPooling1D(pool_size=maxima_longitud_ejemplos))
        print(submodelos[i])
    convoluciones_poolings = Concatenate(submodelos)
    modelo_cnn.add(convoluciones_poolings)
    '''Intento de entrenar con kernels de distintos tamaños al mismo tiempo'''

    # '''
    # Capa Conv1D:
    # Formato de entrada: (maxima_longitud_ejemplos, dimension_embedding)
    # Cantidad de parámetros: (maxima_longitud_ejemplos*DIMENSION_KERNEL*CANTIDAD_FILTROS)+CANTIDAD_FILTROS
    # Formato de salida: (1, maxima_longitud_ejemplos, CANTIDAD_FILTROS)
    # '''
    # modelo_cnn.add(Conv1D(input_shape=formato_entrada,
    #                       filters=CANTIDAD_FILTROS,
    #                       kernel_size=DIMENSION_KERNEL,
    #                       padding='same',
    #                       bias_initializer='random_uniform'))
    # '''
    # Capa MaxPooling1D:
    # Formato de entrada: (1, maxima_longitud_ejemplos)
    # Cantidad de parámetros: 0
    # Formato de salida: (1, 1, CANTIDAD_FILTROS)
    # '''
    # modelo_cnn.add(MaxPooling1D(pool_size=maxima_longitud_ejemplos))
    
    modelo_cnn.add(Dropout(PORCENTAJE_DROPEO))
    modelo_cnn.add(Flatten())

    modelo_cnn.add(Dropout(PORCENTAJE_DROPEO))
    modelo_cnn.add(Dense(NEURONAS_OCULTAS, activation = ACTIVACION_OCULTA))

    modelo_cnn.add(Dropout(PORCENTAJE_DROPEO))
    modelo_cnn.add(Dense(NEURONAS_SALIDA, activation = ACTIVACION_SALIDA))

    modelo_cnn.compile(optimizer = OPTIMIZADOR,
                    loss = FUNCION_ERROR,
                    metrics = ['accuracy'])

    parada_temprana = [EarlyStopping(monitor = 'val_loss',
                                    patience = 0,
                                    verbose = 2,
                                    mode = 'auto')]
                                    
    modelo_cnn.summary()

    print("Particion: {}/{}".format(i+1,k))

    registro = modelo_cnn.fit(x_entrenamiento[folds_dict[i][0]],
                              y_entrenamiento[folds_dict[i][0]],
                              epochs = CANTIDAD_EPOCAS,
                              # callbacks = parada_temprana,
                              validation_data = (x_entrenamiento[folds_dict[i][1]], y_entrenamiento[folds_dict[i][1]]),
                              verbose = 1,
                              batch_size = DIMENSION_BACHA)

    # Resultados de la última época del modelo
    acierto_entrenamiento = registro.history['accuracy'][-1]
    acierto_validacion = registro.history['val_accuracy'][-1]

    resultados_finales.append([acierto_entrenamiento,
                               acierto_validacion])
                            #    acierto_prueba,

    suma_acierto_entrenamiento += acierto_entrenamiento
    suma_acierto_validacion += acierto_validacion
    # suma_acierto_prueba += acierto_prueba
    # Fin del for de particiones

promedio_acierto_entrenamiento = suma_acierto_entrenamiento/k
promedio_acierto_validacion = suma_acierto_validacion/k
# promedio_acierto_prueba = suma_acierto_prueba/k

# Detalles de ejecución
print("Características de los datos de entrada:")
print("\tCantidad de ejemplos: {}".format(cantidad_ejemplos))
print("\tDimension del embedding: {}".format(dimension_embedding))
print("\tLongitud de los ejemplos: {}".format(maxima_longitud_ejemplos))
print("\tTop de palabras frecuentes utilizadas: {}".format(top_palabras_frecuentes))
print("Parámetros:")
print("\tDropout: {}".format(PORCENTAJE_DROPEO))
# print("\tCantidad de convoluciones: {}".format(CANTIDAD_CONVOLUCIONES))
print("\tCantidad de filtros: {}".format(CANTIDAD_FILTROS))
print("\tDimensión de los kernels: {}".format(DIMENSION_KERNEL))
print("\tNeuronas en la capa oculta: {}".format(NEURONAS_OCULTAS))
print("\tActivación en la capa oculta: {}".format(ACTIVACION_OCULTA))
print("\tNeuronas en la capa de salida: {}".format(NEURONAS_SALIDA))
print("\tActivación en la capa de salida: {}".format(ACTIVACION_SALIDA))
print("\tOptimizador: {}".format(OPTIMIZADOR))
print("\tLoss function: {}".format(FUNCION_ERROR))
# print("\tVelocidad de aprendizaje: {}".format(VELOCIDAD_APRENDIZAJE))
print("\tCantidad de épocas: {}".format(CANTIDAD_EPOCAS))
print("\tDimensión batch: {}".format(DIMENSION_BACHA))
print("\tCantidad de particiones: {}".format(k))
print("Resultados del entrenamiento:")
print("\tAcierto en el entrenamiento promedio: {}".format(promedio_acierto_entrenamiento))
for i in range(0, len(resultados_finales), 1):
    print("\t\tAcierto en el entrenamiento en la partición {}: {}".format(i+1, resultados_finales[i][0]))
print("\tAcierto en la validación promedio: {}".format(promedio_acierto_validacion))
for i in range(0, len(resultados_finales), 1):
    print("\t\tAcierto en la validación en la partición {}: {}".format(i+1, resultados_finales[i][1]))
# print("\tAcierto en la prueba promedio: {}".format(promedio_acierto_prueba))
# for i in range(0, len(resultados_finales), 1):
#     print("\t\tAcierto en la prueba en la partición {}: {}".format(i+1, resultados_finales[i][2]))

# fa.graficas(registro)

# Evaluación del modelo
# prueba_error,prueba_acierto = model.evaluate(x_prueba,y_prueba)

# Hacer predicciones
# predicciones = model.predict(x_prueba)

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