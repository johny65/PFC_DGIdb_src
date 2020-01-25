# Paquetes
from __future__ import absolute_import, division, print_function, unicode_literals # Compatibilidad entre Python 2 y 3
import numpy as np
from redes_neuronales_preprocesamiento import cargar_ejemplos, cargar_interacciones # Carga de ejemplos
from keras.models import Sequential, Model
from keras.layers import GRU, LSTM, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.metrics import roc_curve, auc
import statistics
import funciones_auxiliares as fa
import math
import random
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
top_palabras_frecuentes = 300
maxima_longitud_ejemplos = 1000
longitud_palabras_mayor_a = 3
cantidad_ejemplos_sin_interaccion = 1030 # old: 3144; new: 2944
ejemplos_cantidad = 8500

# maxima_longitud_ejemplos: 300 -> top_palabras_frecuentes: 64
# maxima_longitud_ejemplos: 500 -> top_palabras_frecuentes: 103
# maxima_longitud_ejemplos: 1000 -> top_palabras_frecuentes: 552
# maxima_longitud_ejemplos: 1500 -> top_palabras_frecuentes: 790
# maxima_longitud_ejemplos: 2000 -> top_palabras_frecuentes: 1112
# maxima_longitud_ejemplos: 2500 -> top_palabras_frecuentes: 1298
# maxima_longitud_ejemplos: 3000 -> top_palabras_frecuentes: 1421

''' Parámetros del modelo '''
PARTICIONES = 5 # Número de particiones para la validación cruzada
PORCENTAJE_DROPEO = 0.1 # Pone en 0 el #% de los datos aleatoriamente
UNIDADES_LSTM = 100
CANTIDAD_EPOCAS = 100
PORCENTAJE_VALIDACION = 0.2
PORCENTAJE_PRUEBA = 0.2
DIMENSION_BATCH = 32
ACTIVACION_OCULTA = 'relu'
ACTIVACION_SALIDA = 'softmax'
OPTIMIZADOR = "adam" # 'Adam(lr=VELOCIDAD_APRENDIZAJE)'
FUNCION_ERROR = 'categorical_crossentropy'
MODELO_FINAL = False
''' --------------------- '''

(x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = cargar_ejemplos(etiquetas_neural_networks_ruta,
                                                                           ejemplos_directorio,
                                                                           out_interacciones_ruta,
                                                                           embeddings_file=embeddings_file,
                                                                           top_palabras=top_palabras_frecuentes,
                                                                           max_longitud=maxima_longitud_ejemplos,
                                                                           incluir_sin_interacciones = True,
                                                                           sin_interaccion_a_incluir = cantidad_ejemplos_sin_interaccion, # 2944
                                                                           randomize=True,
                                                                           porcentaje_test=PORCENTAJE_PRUEBA,
                                                                           ejemplos_cantidad=ejemplos_cantidad)

cantidad_ejemplos = x_entrenamiento.shape[0]
cantidad_ejemplos_entrenamiento = 0
cantidad_ejemplos_validacion = 0
cantidad_ejemplos_prueba = x_prueba.shape[0]
dimension_embedding = x_entrenamiento.shape[2] # Columnas
cantidad_clases = y_entrenamiento.shape[1]
NEURONAS_SALIDA = cantidad_clases
NEURONAS_OCULTAS = NEURONAS_SALIDA
interacciones_lista = cargar_interacciones(out_interacciones_ruta, True)

if MODELO_FINAL: # Entrenar modelo final
    print("Entrenando modelo final...")
    # x_entrenamiento = np.concatenate((x_entrenamiento, x_prueba))
    # y_entrenamiento = np.concatenate((y_entrenamiento, y_prueba))

    # modelo_rnn = Sequential()

    # formato_entrada = (maxima_longitud_ejemplos, dimension_embedding)
    
    # modelo_rnn.add(LSTM(units=UNIDADES_LSTM,
    #                     dropout=0.0,
    #                     input_shape=formato_entrada))
    
    # # No se hace Flatten()

    # # modelo_rnn.add(Dropout(PORCENTAJE_DROPEO))
    # # modelo_rnn.add(Dense(NEURONAS_OCULTAS, activation = ACTIVACION_OCULTA))

    # modelo_rnn.add(Dropout(PORCENTAJE_DROPEO))
    # modelo_rnn.add(Dense(NEURONAS_SALIDA, activation = ACTIVACION_SALIDA))

    # modelo_rnn.compile(optimizer = OPTIMIZADOR,
    #                 loss = FUNCION_ERROR,
    #                 metrics = ['accuracy'])

    # parada_temprana = [EarlyStopping(monitor = 'loss',
    #                                 patience = 0,
    #                                 verbose = 2,
    #                                 mode = 'auto')]
                                    
    # modelo_rnn.summary()

    # registro = modelo_rnn.fit(x=x_entrenamiento,
    #                           y=y_entrenamiento,
    #                           epochs = CANTIDAD_EPOCAS,
    #                           callbacks = parada_temprana,
    #                           verbose = 1,
    #                           batch_size = DIMENSION_BATCH)

    # # Guardado del modelo
    # modelo_ruta = "rnn.h5"
    # modelo_rnn.save(modelo_ruta)
    # print("Modelo final entrenado y guardado")
else: # Análisis del modelo
    folds_dict = fa.kfolding(PARTICIONES, cantidad_ejemplos, PORCENTAJE_VALIDACION) # Se obtienen las particiones para realizar la validación cruzada

    suma_acierto_entrenamiento = 0
    suma_acierto_validacion = 0
    suma_acierto_prueba = 0

    areas_roc = list()
    resultados_finales = list()

    for i in range(0, PARTICIONES, 1):
        cantidad_ejemplos_entrenamiento = len(x_entrenamiento[folds_dict[i][0]])
        cantidad_ejemplos_validacion = len(x_entrenamiento[folds_dict[i][1]])

        ''' Arquitectura del modelo '''
        modelo_rnn = Sequential()

        formato_entrada = (maxima_longitud_ejemplos, dimension_embedding)

        modelo_rnn.add(LSTM(units=UNIDADES_LSTM,
                            dropout=PORCENTAJE_DROPEO,
                            input_shape=formato_entrada))

        # modelo_rnn.GRU(units=UNIDADES_LSTM,
        #     activation='tanh',
        #     recurrent_activation='sigmoid',
        #     dropout=0.0,
        #     recurrent_dropout=0.0)

        modelo_rnn.add(Dense(NEURONAS_OCULTAS, activation = ACTIVACION_OCULTA))
        modelo_rnn.add(Dropout(PORCENTAJE_DROPEO))
        modelo_rnn.add(Dense(NEURONAS_OCULTAS, activation = ACTIVACION_OCULTA))
        modelo_rnn.add(Dropout(PORCENTAJE_DROPEO))
        modelo_rnn.add(Dense(NEURONAS_SALIDA, activation = ACTIVACION_SALIDA))

        # Se guarda la arquitectura del modelo en un archivo de imagen
        plot_model(modelo_rnn, to_file="modelo_rnn_{}_arquitectura.png".format(i+1))
        ''' Arquitectura del modelo '''

        modelo_rnn.compile(optimizer=OPTIMIZADOR,
                           loss=FUNCION_ERROR,
                           metrics=['accuracy'])

        # Callbacks
        bajar_velocidad = ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.1,
                                            patience=1, # 10
                                            verbose=0,
                                            mode='auto',
                                            min_delta=0.0001,
                                            cooldown=0,
                                            min_lr=0)

        parada_temprana = EarlyStopping(monitor='val_loss',
                                        patience=2, # 0
                                        verbose=2,
                                        mode='auto')
                                        
        modelo_rnn.summary()

        print("Particion: {}/{}".format(i+1,PARTICIONES))

        registro = modelo_rnn.fit(x=x_entrenamiento[folds_dict[i][0]],
                                  y=y_entrenamiento[folds_dict[i][0]],
                                  epochs = CANTIDAD_EPOCAS,
                                  callbacks = [parada_temprana, bajar_velocidad],
                                  validation_data = (x_entrenamiento[folds_dict[i][1]], y_entrenamiento[folds_dict[i][1]]),
                                  verbose = 1,
                                  batch_size = DIMENSION_BATCH)

        _, acierto_prueba = modelo_rnn.evaluate(x_prueba, y_prueba)
        y_prediccion = modelo_rnn.predict(x_prueba)
        razon_falsos_positivos = dict()
        razon_verdaderos_positivos = dict()
        area_bajo_curva_roc = dict()
        for j in range(0, cantidad_clases, 1):
            razon_falsos_positivos[j], razon_verdaderos_positivos[j], _ = roc_curve(y_prueba[:, j], y_prediccion[:, j])
            area_bajo_curva_roc[j] = auc(razon_falsos_positivos[j], razon_verdaderos_positivos[j])
        areas_roc.append(area_bajo_curva_roc)

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

   # Medias
    promedio_acierto_entrenamiento = suma_acierto_entrenamiento/PARTICIONES
    promedio_acierto_validacion = suma_acierto_validacion/PARTICIONES
    promedio_acierto_prueba = suma_acierto_prueba/PARTICIONES

    # Desvíos estándar
    resultados_finales = np.asarray(resultados_finales)
    desvio_acierto_entrenamiento = statistics.stdev(resultados_finales[:, 0])
    desvio_acierto_validacion = statistics.stdev(resultados_finales[:, 1])
    desvio_acierto_prueba = statistics.stdev(resultados_finales[:, 2])

    # AUC's ROC
    promedios_desvios_auc_roc = dict()
    for i in range(0, cantidad_clases, 1):
        lista = list()
        for j in range(0, PARTICIONES, 1):
            area = areas_roc[j][i]
            lista.append(area)
        media = statistics.mean(lista)
        desvio = statistics.stdev(lista)
        promedios_desvios_auc_roc[i] = [media, desvio]
    
    # Detalles de ejecución
    print("Características de los datos de entrada:")
    # print("\tCantidad de ejemplos cargados: {}".format(2944 + cantidad_ejemplos_sin_interaccion))
    # print("\tCantidad de ejemplos con la etiqueta sin_interacción: {}".format(cantidad_ejemplos_sin_interaccion))
    print("\tCantidad de ejemplos utilizados para entrenar: {}".format(cantidad_ejemplos_entrenamiento))
    print("\tCantidad de ejemplos utilizados para validar: {}".format(cantidad_ejemplos_validacion))
    print("\tCantidad de ejemplos utilizados para probar: {}".format(cantidad_ejemplos_prueba))
    print("\tTop de palabras frecuentes utilizadas: {}".format(top_palabras_frecuentes))
    print("\tSolo palabras con longitud mayor a: {}".format(longitud_palabras_mayor_a))
    print("\tLongitud de los ejemplos (filas): {}".format(maxima_longitud_ejemplos))
    print("\tDimension del embedding (columnas): {}".format(dimension_embedding))

    print("Parámetros:")
    print("\tCantidad de unidades LSTM: {}".format(UNIDADES_LSTM))
    print("\tCantidad de épocas: {}".format(CANTIDAD_EPOCAS))
    print("\tCantidad de particiones: {}".format(PARTICIONES))
    print("\tDropout: {}".format(PORCENTAJE_DROPEO))
    print("\tNeuronas en la capa oculta: {}".format(NEURONAS_OCULTAS))
    print("\tActivación en la capa oculta: {}".format(ACTIVACION_OCULTA))
    print("\tNeuronas en la capa de salida: {}".format(NEURONAS_SALIDA))
    print("\tActivación en la capa de salida: {}".format(ACTIVACION_SALIDA))
    print("\tOptimizador: {}".format(OPTIMIZADOR))
    print("\tLoss function: {}".format(FUNCION_ERROR))
    print("\tDimensión batch: {}".format(DIMENSION_BATCH))
    
    print("Resultados del entrenamiento:")
    print("\tAcierto en el entrenamiento - media: {}".format(promedio_acierto_entrenamiento))
    print("\tAcierto en el entrenamiento - desvío: {}".format(desvio_acierto_entrenamiento))
    for i in range(0, len(resultados_finales), 1):
        print("\t\tAcierto en el entrenamiento en la partición {}: {}".format(i+1, resultados_finales[i][0]))

    print("\tAcierto en la validación - media: {}".format(promedio_acierto_validacion))
    print("\tAcierto en el validación - desvío: {}".format(desvio_acierto_validacion))
    for i in range(0, len(resultados_finales), 1):
        print("\t\tAcierto en la validación en la partición {}: {}".format(i+1, resultados_finales[i][1]))

    print("\tAcierto en la prueba - media: {}".format(promedio_acierto_prueba))
    print("\tAcierto en el prueba - desvío: {}".format(desvio_acierto_prueba))
    for i in range(0, len(resultados_finales), 1):
        print("\t\tAcierto en la prueba en la partición {}: {}".format(i+1, resultados_finales[i][2]))

    print("\tÁreas bajo la curva ROC para las distintas clases:")
    for i in range(0, len(promedios_desvios_auc_roc), 1):
        print("\t\tMedia y desvío AUC ROC de la clase {}: {}".format(interacciones_lista[i], promedios_desvios_auc_roc[i]))

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