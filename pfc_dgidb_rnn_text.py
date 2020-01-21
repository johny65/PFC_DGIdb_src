''' Paquetes '''
from __future__ import absolute_import, division, print_function, unicode_literals # Compatibilidad entre Python 2 y 3
import numpy as np
from redes_neuronales_preprocesamiento import cargar_ejemplos # Carga de ejemplos
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
import funciones_auxiliares as fa
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Para utilizar CPU en lugar de GPU

''' Carga de datos '''
etiquetas_neural_networks_ruta = "etiquetas_neural_networks.csv"
ejemplos_directorio = "replaced_new"
out_interacciones_ruta = "interacciones_lista.txt"
embeddings_file = "glove.6B.50d.txt"
# embeddings_file = "glove.6B.100d.txt"
# embeddings_file = "glove.6B.200d.txt"
# embeddings_file = "glove.6B.300d.txt"
top_palabras_frecuentes = 100
maxima_longitud_ejemplos = 500 # Filas

# maxima_longitud_ejemplos: 300 -> top_palabras_frecuentes: 64
# maxima_longitud_ejemplos: 500 -> top_palabras_frecuentes: 103
# maxima_longitud_ejemplos: 1000 -> top_palabras_frecuentes: 552
# maxima_longitud_ejemplos: 1500 -> top_palabras_frecuentes: 790
# maxima_longitud_ejemplos: 2000 -> top_palabras_frecuentes: 1112
# maxima_longitud_ejemplos: 2500 -> top_palabras_frecuentes: 1298
# maxima_longitud_ejemplos: 3000 -> top_palabras_frecuentes: 1421

'''VARIABLES GLOBALES'''
CANTIDAD_EJEMPLOS_SIN_INTERACCION = 2944
k = 10 # Número de particiones para la validación cruzada
CANTIDAD_EPOCAS = 100
PORCENTAJE_PRUEBA = 0.2
PORCENTAJE_VALIDACION = 0.2
DIMENSION_LSTM = 100
PORCENTAJE_DROPEO = 0.4
DIMENSION_BATCH = 32
NEURONAS_OCULTAS = 30
ACTIVACION_OCULTA = "relu"
ACTIVACION_SALIDA = "softmax"
FUNCION_ERROR = "categorical_crossentropy"
OPTIMIZADOR = "adam"
MODELO_FINAL = False
'''------------------'''

(x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = cargar_ejemplos(etiquetas_neural_networks_ruta,
                                                           ejemplos_directorio,
                                                           out_interacciones_ruta,
                                                           embeddings_file=embeddings_file,
                                                           top_palabras=top_palabras_frecuentes,
                                                           max_longitud=maxima_longitud_ejemplos,
                                                           incluir_sin_interacciones = False,
                                                           sin_interaccion_a_incluir = CANTIDAD_EJEMPLOS_SIN_INTERACCION, # 2944
                                                           randomize=True,
                                                           porcentaje_test=PORCENTAJE_PRUEBA)

cantidad_ejemplos = x_entrenamiento.shape[0]
dimension_embedding = x_entrenamiento.shape[2] # Columnas
NEURONAS_SALIDA = y_entrenamiento.shape[1] # Cantidad de clases

if MODELO_FINAL: # Entrenar modelo final
    print("Entrenando modelo final...")
    x_entrenamiento = np.concatenate((x_entrenamiento, x_prueba))
    y_entrenamiento = np.concatenate((y_entrenamiento, y_prueba))

    modelo_rnn = Sequential()

    formato_entrada = (maxima_longitud_ejemplos, dimension_embedding)
    
    modelo_rnn.add(LSTM(units=DIMENSION_LSTM,
                        dropout=0.0,
                        input_shape=formato_entrada))
    
    # No se hace Flatten()

    # modelo_rnn.add(Dropout(PORCENTAJE_DROPEO))
    # modelo_rnn.add(Dense(NEURONAS_OCULTAS, activation = ACTIVACION_OCULTA))

    modelo_rnn.add(Dropout(PORCENTAJE_DROPEO))
    modelo_rnn.add(Dense(NEURONAS_SALIDA, activation = ACTIVACION_SALIDA))

    modelo_rnn.compile(optimizer = OPTIMIZADOR,
                    loss = FUNCION_ERROR,
                    metrics = ['accuracy'])

    parada_temprana = [EarlyStopping(monitor = 'loss',
                                    patience = 0,
                                    verbose = 2,
                                    mode = 'auto')]
                                    
    modelo_rnn.summary()

    registro = modelo_rnn.fit(x=x_entrenamiento,
                              y=y_entrenamiento,
                              epochs = CANTIDAD_EPOCAS,
                              callbacks = parada_temprana,
                              verbose = 1,
                              batch_size = DIMENSION_BATCH)

    # Guardado del modelo
    modelo_ruta = "rnn.h5"
    modelo_rnn.save(modelo_ruta)
    print("Modelo final entrenado y guardado")
else: # Análisis del modelo
    folds_dict = fa.kfolding(k, cantidad_ejemplos, PORCENTAJE_VALIDACION) # Se obtienen las particiones para realizar la validación cruzada

    suma_acierto_entrenamiento = 0
    suma_acierto_validacion = 0
    suma_acierto_prueba = 0

    resultados_finales = list()

    for i in range(0,k,1):
        modelo_rnn = Sequential()

        formato_entrada = (maxima_longitud_ejemplos, dimension_embedding)

        modelo_rnn.add(LSTM(units=DIMENSION_LSTM,
                            dropout=0.0,
                            input_shape=formato_entrada))
        
        # No se hace Flatten()

        # modelo_rnn.add(Dropout(PORCENTAJE_DROPEO))
        # modelo_rnn.add(Dense(NEURONAS_OCULTAS, activation = ACTIVACION_OCULTA))

        modelo_rnn.add(Dropout(PORCENTAJE_DROPEO))
        modelo_rnn.add(Dense(NEURONAS_SALIDA, activation = ACTIVACION_SALIDA))

        modelo_rnn.compile(optimizer = OPTIMIZADOR,
                        loss = FUNCION_ERROR,
                        metrics = ['accuracy'])

        parada_temprana = [EarlyStopping(monitor = 'val_loss',
                                        patience = 0,
                                        verbose = 2,
                                        mode = 'auto')]
                                        
        modelo_rnn.summary()

        print("Particion: {}/{}".format(i+1,k))

        registro = modelo_rnn.fit(x_entrenamiento[folds_dict[i][0]],
                                y_entrenamiento[folds_dict[i][0]],
                                epochs = CANTIDAD_EPOCAS,
                                callbacks = parada_temprana,
                                validation_data = (x_entrenamiento[folds_dict[i][1]], y_entrenamiento[folds_dict[i][1]]),
                                verbose = 1,
                                batch_size = DIMENSION_BATCH)

        _, acierto_prueba = modelo_rnn.evaluate(x_prueba, y_prueba)

        # Resultados de la última época del modelo
        acierto_entrenamiento = registro.history['accuracy'][-1]
        acierto_validacion = registro.history['val_accuracy'][-1]

        resultados_finales.append([acierto_entrenamiento,
                                acierto_validacion,
                                acierto_prueba])

        suma_acierto_entrenamiento += acierto_entrenamiento
        suma_acierto_validacion += acierto_validacion
        suma_acierto_prueba += acierto_prueba
        # Fin del for de particiones

    promedio_acierto_entrenamiento = suma_acierto_entrenamiento/k
    promedio_acierto_validacion = suma_acierto_validacion/k
    promedio_acierto_prueba = suma_acierto_prueba/k

    # Detalles de ejecución
    print("Características de los datos de entrada:")
    print("\tCantidad total de ejemplos cargados: {}".format(cantidad_ejemplos))
    print("\tCantidad de ejemplos de entrenamiento: {}".format(cantidad_ejemplos))
    print("Parámetros:")
    print("\tDropout: {}".format(PORCENTAJE_DROPEO))
    print("\tDimensión LSTM: {}".format(DIMENSION_LSTM))
    print("\tNeuronas en la capa oculta: {}".format(NEURONAS_OCULTAS))
    print("\tFunción de activación en la capa oculta: {}".format(ACTIVACION_OCULTA))
    print("\tNeuronas en la capa de salida: {}".format(NEURONAS_SALIDA))
    print("\tFunción de activación en la capa de salida: {}".format(ACTIVACION_SALIDA))
    print("\tFunción de error: {}".format(FUNCION_ERROR))
    print("\tOptimizador: {}".format(OPTIMIZADOR))
    print("\tCantidad de épocas: {}".format(CANTIDAD_EPOCAS))
    print("\tDimensión batch: {}".format(DIMENSION_BATCH))
    print("\tCantidad de particiones: {}".format(k))
    print("Resultados del entrenamiento:")
    print("\tAcierto en el entrenamiento promedio: {}".format(promedio_acierto_entrenamiento))
    for i in range(0, len(resultados_finales), 1):
        print("\t\tAcierto en el entrenamiento en la partición {}: {}".format(i+1, resultados_finales[i][0]))
    print("\tAcierto en la validación promedio: {}".format(promedio_acierto_validacion))
    for i in range(0, len(resultados_finales), 1):
        print("\t\tAcierto en la validación en la partición {}: {}".format(i+1, resultados_finales[i][1]))
    print("\tAcierto en la prueba promedio: {}".format(promedio_acierto_prueba))
    for i in range(0, len(resultados_finales), 1):
        print("\t\tAcierto en la prueba en la partición {}: {}".format(i+1, resultados_finales[i][2]))