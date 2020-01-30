# Paquetes
from __future__ import absolute_import, division, print_function, unicode_literals # Compatibilidad entre Python 2 y 3
import numpy as np
from redes_neuronales_preprocesamiento import cargar_ejemplos, cargar_interacciones, otro_cargar_ejemplos # Carga de ejemplos
from keras.models import Sequential, Model, load_model
from keras.layers import Conv1D, MaxPooling1D, Concatenate, Dropout, Flatten, Dense, concatenate, Input, BatchNormalization, Activation, Embedding, LSTM, GRU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam,SGD # SGD: Gradiente descendiente
from keras.utils import plot_model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as pplt
import statistics
import funciones_auxiliares as fa
import math
import random
import math
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Para utilizar CPU en lugar de GPU

''' Carga de datos '''
etiquetas_neural_networks_ruta = "etiquetas_neural_networks_3.csv"
out_interacciones_ruta = "interacciones_lista.txt"
excluir_interacciones_lista = []
ejemplos_directorio = "replaced_new"
# embeddings_file = "glove.6B.50d.txt"
# embeddings_file = "glove.6B.100d.txt"
# embeddings_file = "glove.6B.200d.txt"
# embeddings_file = "glove.6B.300d.txt"
dimension_embedding = 12
maxima_longitud_ejemplos = 100 # Longitud promedio de los ejemplos: 6500
top_palabras_considerar = 50

cantidad_ejemplos_clase_mayor = 1031
clases = 27
ejemplos_cantidad = cantidad_ejemplos_clase_mayor*clases # Cantidad total de ejemplos a cargar

balancear = False
vocabulario_bool = False
secuencias_bool = False
particiones_bool = False
embeddings_bool = False

''' Parámetros del modelo '''
PARTICIONES = 5 # Número de particiones para la validación cruzada
REPETICIONES = 1 # Número de repeticiones de la validación cruzada
PORCENTAJE_DROPEO = 0.4 # Pone en 0 el #% de los datos aleatoriamente
CANTIDAD_UNIDADES = 100
CAPAS_OCULTAS = 2
CANTIDAD_EPOCAS = 100
PORCENTAJE_VALIDACION = 0.2
PORCENTAJE_PRUEBA = 0.2
DIMENSION_BATCH = 32
ACTIVACION_OCULTA = 'relu'
ACTIVACION_SALIDA = 'softmax'
OPTIMIZADOR = "adam"
FUNCION_ERROR = 'categorical_crossentropy'
MODELO_FINAL = False
''' --------------------- '''

# (x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = cargar_ejemplos(etiquetas_neural_networks_ruta,
#                                                                            ejemplos_directorio,
#                                                                            out_interacciones_ruta,
#                                                                            embeddings_file=embeddings_file,
#                                                                            top_palabras=top_palabras_frecuentes,
#                                                                            max_longitud=maxima_longitud_ejemplos,
#                                                                            incluir_sin_interacciones = True,
#                                                                            sin_interaccion_a_incluir = cantidad_ejemplos_sin_interaccion, # 2944
#                                                                            randomize=False,
#                                                                            porcentaje_test=PORCENTAJE_PRUEBA,
#                                                                            ejemplos_cantidad=ejemplos_cantidad,
#                                                                            vocabulario_global=True)

(x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba), top_palabras_frecuentes, vocabulario = otro_cargar_ejemplos(etiquetas_neural_networks_ruta,
                                                                                                                      out_interacciones_ruta,
                                                                                                                      excluir_interacciones_lista,
                                                                                                                      ejemplos_cantidad,
                                                                                                                      PORCENTAJE_PRUEBA,
                                                                                                                      ejemplos_directorio,
                                                                                                                      maxima_longitud_ejemplos,
                                                                                                                      top_palabras_considerar,
                                                                                                                      balancear,
                                                                                                                      vocabulario_bool,
                                                                                                                      secuencias_bool,
                                                                                                                      particiones_bool,
                                                                                                                      embeddings_bool)

cantidad_ejemplos = x_entrenamiento.shape[0]
cantidad_ejemplos_entrenamiento = 0
cantidad_ejemplos_validacion = 0
cantidad_ejemplos_prueba = x_prueba.shape[0]
cantidad_clases = y_entrenamiento.shape[1]
NEURONAS_SALIDA = cantidad_clases
NEURONAS_OCULTAS = NEURONAS_SALIDA*3
interacciones_lista = cargar_interacciones(out_interacciones_ruta, True)

# print("Cargando embeddings pre-entrenados.")
# embeddings_dict = dict()
# with open(embeddings_file, encoding="utf8") as embeddings:
#     for linea in embeddings:
#         linea = linea.split()
#         palabra = linea[0]
#         embedding = np.asarray(linea[1:] + [0, 0], dtype='float64')
#         embeddings_dict[palabra] = embedding
# print("Embeddings pre-entrenados cargados.")

# print("Generando matriz de embeddings.")
# gen_emb = np.zeros((1, dimension_embedding))
# gen_emb[0, dimension_embedding-2] = 1
# droga_emb = np.zeros((1, dimension_embedding))
# droga_emb[0, dimension_embedding-1] = 1
# embeddings_matriz = np.zeros((top_palabras_frecuentes, dimension_embedding))
# for palabra, secuencia in vocabulario.word_index.items():
#     embedding_vector = embeddings_dict.get(palabra)
#     if embedding_vector is not None:
#         embeddings_matriz[i] = embedding_vector
#     elif palabra.startswith("xxxg") and palabra.endswith("xxx"):
#         embeddings_matriz[i] = gen_emb
#     elif palabra.startswith("xxxd") and palabra.endswith("xxx"):
#         embeddings_matriz[i] = droga_emb
# print("Matriz de embeddings generada.")

if MODELO_FINAL: # Entrenar modelo final
    print("Entrenando modelo final...")
else: # Análisis del modelo
    areas_roc = list()
    resultados_finales = list()
    for r in range(0, REPETICIONES, 1): # Número de veces que se repite la validación cruzada
        # Aleatoriza los ejemplos
        seed = random.random()
        random.seed(seed)
        indices_aleatorios = np.arange(len(x_entrenamiento))
        random.shuffle(indices_aleatorios)
        x_entrenamiento = x_entrenamiento[indices_aleatorios]
        y_entrenamiento = y_entrenamiento[indices_aleatorios]

        folds_dict = fa.kfolding(PARTICIONES, cantidad_ejemplos, PORCENTAJE_VALIDACION) # Se obtienen las particiones para realizar la validación cruzada

        for i in range(0, PARTICIONES, 1):
            cantidad_ejemplos_entrenamiento = len(x_entrenamiento[folds_dict[i][0]])
            cantidad_ejemplos_validacion = len(x_entrenamiento[folds_dict[i][1]])

            ''' Arquitectura del modelo '''
            modelo_rnn = Sequential()

            # Formato de entrada
            formato_entrada = (maxima_longitud_ejemplos, dimension_embedding)

            # Capa recurrente
            modelo_rnn.add(LSTM(units=CANTIDAD_UNIDADES,
                                dropout=PORCENTAJE_DROPEO,
                                input_shape=formato_entrada))
            
            # modelo_rnn.add(GRU(units=CANTIDAD_UNIDADES,
            #                    dropout=PORCENTAJE_DROPEO,
            #                    input_shape=formato_entrada))

            # Capas ocultas
            for j in range(0, CAPAS_OCULTAS, 1):
                modelo_rnn.add(Dense(NEURONAS_OCULTAS, use_bias=False)) 
                modelo_rnn.add(BatchNormalization())
                modelo_rnn.add(Activation(ACTIVACION_OCULTA))
                modelo_rnn.add(Dropout(PORCENTAJE_DROPEO))

            # Capa de salida
            modelo_rnn.add(Dense(NEURONAS_SALIDA, activation=ACTIVACION_SALIDA))

            # Se guarda la arquitectura del modelo en un archivo de imagen
            plot_model(modelo_rnn, to_file="modelo_rnn_arquitectura.png")
            ''' Arquitectura del modelo '''

            modelo_rnn.compile(optimizer=OPTIMIZADOR,
                               loss=FUNCION_ERROR,
                               metrics=['accuracy'])

            # Callbacks
            bajar_velocidad = ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.1,
                                                patience=2, # 10
                                                verbose=1,
                                                mode='min',
                                                min_delta=0.0001,
                                                cooldown=0,
                                                min_lr=0)

            parada_temprana_val_loss = EarlyStopping(monitor='val_loss',
                                                     patience=4,
                                                     mode='min',
                                                     verbose=1)                                                     

            modelo_punto_de_control = ModelCheckpoint("mejor_modelo_cnn_{}.h5".format((r+1)*(i+1)),
                                                      monitor="val_accuracy",
                                                      mode="max",
                                                      save_best_only=True,
                                                      verbose=1)
                                            
            modelo_rnn.summary()

            print("Repetición: {}/{} - Particion: {}/{}".format(r+1, REPETICIONES, i+1, PARTICIONES))

            registro = modelo_rnn.fit(x=x_entrenamiento[folds_dict[i][0]],
                                      y=y_entrenamiento[folds_dict[i][0]],
                                      epochs=CANTIDAD_EPOCAS,
                                      callbacks=[parada_temprana_val_loss, bajar_velocidad, modelo_punto_de_control],
                                      validation_data=(x_entrenamiento[folds_dict[i][1]], y_entrenamiento[folds_dict[i][1]]),
                                      verbose=1,
                                      batch_size=DIMENSION_BATCH)

            mejor_modelo_rnn = load_model("mejor_modelo_rnn_{}.h5".format((r+1)*(i+1)))

            _, acierto_entrenamiento = mejor_modelo_rnn.evaluate(x_entrenamiento[folds_dict[i][0]], y_entrenamiento[folds_dict[i][0]])
            _, acierto_validacion = mejor_modelo_rnn.evaluate(x_entrenamiento[folds_dict[i][1]], y_entrenamiento[folds_dict[i][1]])
            _, acierto_prueba = mejor_modelo_rnn.evaluate(x_prueba, y_prueba)

            print("Acierto en el entrenamiento: {}%".format("%.2f" % (acierto_entrenamiento*100)))
            print("Acierto en la validación: {}%".format("%.2f" % (acierto_validacion*100)))
            print("Acierto en la prueba: {}%".format("%.2f" % (acierto_prueba*100)))

            resultados_finales.append([acierto_entrenamiento,
                                       acierto_validacion,
                                       acierto_prueba])

            y_prediccion = modelo_rnn.predict(x_prueba)
            razon_falsos_positivos = dict()
            razon_verdaderos_positivos = dict()
            area_bajo_curva_roc = dict()
            for j in range(cantidad_clases):
                razon_falsos_positivos[j], razon_verdaderos_positivos[j], _ = roc_curve(y_prueba[:, j], y_prediccion[:, j])
                area_bajo_curva_roc[j] = auc(razon_falsos_positivos[j], razon_verdaderos_positivos[j])
            areas_roc.append(area_bajo_curva_roc)
            # Fin del for de particiones
    # Fin del for de repeticiones

    resultados_finales = np.asarray(resultados_finales)

    # Medias
    promedio_acierto_entrenamiento = statistics.mean(resultados_finales[:, 0])
    promedio_acierto_validacion = statistics.mean(resultados_finales[:, 1])
    promedio_acierto_prueba = statistics.mean(resultados_finales[:, 2])

    # Desvíos estándar
    desvio_acierto_entrenamiento = statistics.stdev(resultados_finales[:, 0])
    desvio_acierto_validacion = statistics.stdev(resultados_finales[:, 1])
    desvio_acierto_prueba = statistics.stdev(resultados_finales[:, 2])

    # AUC's ROC
    promedios_desvios_auc_roc = dict()
    for i in range(0, cantidad_clases, 1):
        lista = list()
        for j in range(0, PARTICIONES*REPETICIONES, 1):
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
    # print("\tSolo palabras con longitud mayor a: {}".format(longitud_palabras_mayor_a))
    print("\tLongitud de los ejemplos (filas): {}".format(maxima_longitud_ejemplos))
    print("\tDimension del embedding (columnas): {}".format(dimension_embedding))

    print("Parámetros:")
    print("\tDropout: {}".format(PORCENTAJE_DROPEO))
    print("\tCantidad de filtros: {}".format(CANTIDAD_FILTROS))
    print("\tDimensión de los kernels: {}".format(DIMENSION_KERNEL))
    print("\tNeuronas en la capa oculta: {}".format(NEURONAS_OCULTAS))
    print("\tActivación en la capa oculta: {}".format(ACTIVACION_OCULTA))
    print("\tNeuronas en la capa de salida: {}".format(NEURONAS_SALIDA))
    print("\tActivación en la capa de salida: {}".format(ACTIVACION_SALIDA))
    print("\tOptimizador: {}".format(OPTIMIZADOR))
    print("\tLoss function: {}".format(FUNCION_ERROR))
    print("\tCantidad de épocas: {}".format(CANTIDAD_EPOCAS))
    print("\tDimensión batch: {}".format(DIMENSION_BATCH))
    print("\tCantidad de particiones: {}".format(PARTICIONES))

    print("Resultados del entrenamiento:")
    print("\tAcierto en el entrenamiento [media, desvío]: [{}% / {}%]".format("%.2f" % (promedio_acierto_entrenamiento*100), "%.2f" % (100*desvio_acierto_entrenamiento)))
    # for i in range(0, len(resultados_finales), 1):
    #     print("\t\tAcierto en el entrenamiento en la repeteción {}, partición {}: {}".format(r+1, i+1, resultados_finales[i][0]))

    print("\tAcierto en el validación [media, desvío]: [{}% / {}%]".format("%.2f" % (promedio_acierto_validacion*100), "%.2f" % (100*desvio_acierto_validacion)))
    # for i in range(0, len(resultados_finales), 1):
    #     print("\t\tAcierto en la validación en la repeteción {}, partición {}: {}".format(r+1, i+1, resultados_finales[i][1]))

    print("\tAcierto en el prueba [media, desvío]: [{}% / {}%]".format("%.2f" % (promedio_acierto_prueba*100), "%.2f" % (100*desvio_acierto_prueba)))
    # for i in range(0, len(resultados_finales), 1):
    #     print("\t\tAcierto en la prueba en la repeteción {}, partición {}: {}".format(r+1, i+1, resultados_finales[i][2]))

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