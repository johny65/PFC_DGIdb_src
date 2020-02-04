# Paquetes
from __future__ import absolute_import, division, print_function, unicode_literals # Compatibilidad entre Python 2 y 3
import numpy as np
from redes_neuronales_preprocesamiento import cargar_ejemplos, cargar_interacciones, otro_cargar_ejemplos # Carga de ejemplos
from keras.models import Sequential, Model, load_model
from keras.layers import Conv1D, MaxPooling1D, Concatenate, Dropout, Flatten, Dense, concatenate, Input, BatchNormalization, Activation, Embedding, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam,SGD # SGD: Gradiente descendiente
from keras.utils import plot_model
from keras import backend as K
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as pplt
import statistics
import funciones_auxiliares as fa
import math
import random
import math
import os
import pathlib
import pickle
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Para utilizar CPU en lugar de GPU

K.set_floatx('float32')

''' Carga de datos '''
etiquetas_neural_networks_ruta = "etiquetas_neural_networks_4.csv"
out_interacciones_ruta = "interacciones_lista.txt"
excluir_interacciones_lista = []
ejemplos_directorio = "replaced4"
# embeddings_file = "glove.6B.50d.txt"
# embeddings_file = "glove.6B.100d.txt"
# embeddings_file = "glove.6B.200d.txt"
# embeddings_file = "glove.6B.300d.txt"
dimension_embedding = 128 # Recomendado: 8
maxima_longitud_ejemplos = 10000 # Máxima: 157170; Promedio: 5201.110850439883
vocabulario_bool = False
secuencias_bool = False
particiones_bool = False
padtrunc_where = "post"

''' Parámetros del modelo '''
PARTICIONES = 5 # Número de particiones para la validación cruzada
REPETICIONES = 1 # Número de repeticiones de la validación cruzada
PORCENTAJE_DROPEO = 0.4 # Recomendado: 0.4
CANTIDAD_EPOCAS = 100
PORCENTAJE_VALIDACION = 0.2
# Perceptrón
CAPAS_OCULTAS = 1
ACTIVACION_OCULTA = "relu"
ACTIVACION_SALIDA = "softmax"
# Convolución
CANTIDAD_FILTROS = 100 # Recomendado: 35
DIMENSION_KERNEL = list()
for i in range(3, 10, 2):
    DIMENSION_KERNEL.append(i)
DIMENSION_KERNEL = tuple(DIMENSION_KERNEL)
print(DIMENSION_KERNEL)
# Parámetros de compilación
OPTIMIZADOR = "adam"
FUNCION_ERROR = "categorical_crossentropy"
METRICA = "categorical_accuracy" # categorical_accuracy
# Parámetros de entrenamiento
DIMENSION_BATCH = 8

MODELO_FINAL = False
''' --------------------- '''

(x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba), vocabulario = otro_cargar_ejemplos(etiquetas_neural_networks_ruta,
                                                                                            out_interacciones_ruta,
                                                                                            excluir_interacciones_lista,
                                                                                            None,
                                                                                            ejemplos_directorio,
                                                                                            maxima_longitud_ejemplos,
                                                                                            vocabulario_bool,
                                                                                            secuencias_bool,
                                                                                            particiones_bool,
                                                                                            padtrunc_where)

cantidad_ejemplos = x_entrenamiento.shape[0]
cantidad_ejemplos_entrenamiento = 0
cantidad_ejemplos_validacion = 0
cantidad_ejemplos_prueba = x_prueba.shape[0]
cantidad_clases = y_entrenamiento.shape[1]
NEURONAS_SALIDA = cantidad_clases
NEURONAS_OCULTAS = NEURONAS_SALIDA*3
interacciones_lista = cargar_interacciones(out_interacciones_ruta, True)

interacciones_pesos_dict = fa.generar_pesos_clases(etiquetas_neural_networks_ruta,
                                                   excluir_interacciones_lista,
                                                   interacciones_lista)
print(interacciones_pesos_dict)

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

# print("Generando matriz de embeddings.")
# embeddings_matriz = np.zeros((top_palabras_frecuentes, dimension_embedding))
# for palabra, secuencia in vocabulario.word_index.items():
#     embedding_vector = embeddings_dict.wv[palabra]
#     if embedding_vector is not None:
#         embeddings_matriz[i] = embedding_vector
# print("Matriz de embeddings generada.")

if MODELO_FINAL: # Entrenar modelo final
    print("Vacío")
    # # Guardado del modelo
    # modelo_ruta = "cnn.h5"
    # modelo_cnn.save(modelo_ruta)
    # print("Modelo final entrenado y guardado.")
else: # Análisis del modelo
    areas_roc = list()
    resultados_finales = list()

    kfold = StratifiedKFold(n_splits=PARTICIONES, shuffle=True)
    # folds_dict = fa.kfolding(PARTICIONES, cantidad_ejemplos, PORCENTAJE_VALIDACION) # Se obtienen las particiones para realizar la validación cruzada

    y_para_split = [y.tolist().index(1) for y in y_entrenamiento]
    i = 0
    for train_index, val_index in kfold.split(x_entrenamiento, y_para_split):
        i += 1
        cantidad_ejemplos_entrenamiento = len(train_index)
        cantidad_ejemplos_validacion = len(val_index)

        ''' Arquitectura del modelo '''
        formato_entrada = (maxima_longitud_ejemplos,)

        # Capa de entrada
        entrada = Input(formato_entrada)

        # Capa de embedding
        embedding = Embedding(input_dim=len(vocabulario.index_word),
                                output_dim=dimension_embedding,
                                input_length=maxima_longitud_ejemplos)(entrada)
                            #   weights=[embeddings_matriz], # Embeddings de GloVe
                            #   trainable=True)(entrada)

        # dropout_embedding = Dropout(PORCENTAJE_DROPEO)(embedding)

        # Capas de convolución y pooling
        capas = list()
        for j in range(0, len(DIMENSION_KERNEL), 1):
            convolucion = Conv1D(filters=CANTIDAD_FILTROS,
                                kernel_size=DIMENSION_KERNEL[j],
                                padding='same',
                                # padding='valid',
                                # use_bias=False)(entrada)
                                use_bias=False)(embedding) # dropout_embedding
            batch_normalization = BatchNormalization()(convolucion)
            activacion = Activation(ACTIVACION_OCULTA)(batch_normalization)
            # pooling = MaxPooling1D(pool_size=maxima_longitud_ejemplos)(activacion)
            pooling = GlobalMaxPooling1D()(activacion)
            dropout = Dropout(PORCENTAJE_DROPEO)(pooling)
            capas.append(dropout)

        # Concatenación de las convoluciones y poolings
        if len(DIMENSION_KERNEL) > 1:
            convoluciones_poolings = concatenate(capas)
        else:
            convoluciones_poolings = capas[0]

        # Capa de aplanado
        # flatten = Flatten()(convoluciones_poolings)
        
        # Capas ocultas
        ultima_capa = 0
        dense = 0
        for j in range(0, CAPAS_OCULTAS, 1):
            if j == 0:
                dense = Dense(NEURONAS_OCULTAS, use_bias=False)(convoluciones_poolings) # flatten
            else:
                dense = Dense(NEURONAS_OCULTAS, use_bias=False)(ultima_capa)
            batch_normalization = BatchNormalization()(dense)
            activacion = Activation(ACTIVACION_OCULTA)(batch_normalization)
            dropout = Dropout(PORCENTAJE_DROPEO)(activacion)
            ultima_capa = dropout
        
        # Capa de salida
        dense3 = Dense(NEURONAS_SALIDA, activation=ACTIVACION_SALIDA)(ultima_capa)
        modelo_cnn = Model(input=entrada, output=dense3)

        # Se guarda la arquitectura del modelo en un archivo de imagen
        plot_model(modelo_cnn, to_file="modelo_cnn_arquitectura.png")
        ''' Arquitectura del modelo '''

        modelo_cnn.compile(optimizer=OPTIMIZADOR,
                            loss=FUNCION_ERROR,
                            metrics=[METRICA])

        # Callbacks
        bajar_velocidad = ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.1,
                                            patience=2, # 10
                                            verbose=1,
                                            mode='auto',
                                            min_delta=0.0001,
                                            cooldown=0,
                                            min_lr=0)

        parada_temprana_val_loss = EarlyStopping(monitor='val_loss',
                                                    patience=4, # 4
                                                    mode='auto',
                                                    verbose=1)                                                     

        modelo_punto_de_control = ModelCheckpoint("mejor_modelo_cnn_{}.h5".format(i+1),
                                                    monitor="val_categorical_accuracy",
                                                    mode="auto",
                                                    save_best_only=True,
                                                    verbose=1)

        modelo_cnn.summary() # Detalles del modelo

        print("Particion: {}/{}".format(i, PARTICIONES))

        registro = modelo_cnn.fit(x=x_entrenamiento[train_index],
                                    y=y_entrenamiento[train_index],
                                    epochs=CANTIDAD_EPOCAS,
                                    callbacks=[parada_temprana_val_loss, bajar_velocidad, modelo_punto_de_control], #
                                    validation_data=(x_entrenamiento[val_index], y_entrenamiento[val_index]),
                                    verbose=1,
                                    class_weight=interacciones_pesos_dict,
                                    batch_size=DIMENSION_BATCH)

        # pplt.plot(registro.history["loss"], label="Error entrenamiento")
        # pplt.plot(registro.history["val_loss"], label="Error validación")
        # pplt.plot(registro.history["accuracy"], label="Acierto entrenamiento")
        # pplt.plot(registro.history["val_accuracy"], label="Acierto validación")
        # pplt.legend()
        # pplt.show()

        del modelo_cnn
        modelo_cnn = load_model("mejor_modelo_cnn_{}.h5".format(i+1))

        _, acierto_entrenamiento = modelo_cnn.evaluate(x_entrenamiento[train_index], y_entrenamiento[train_index])
        _, acierto_validacion = modelo_cnn.evaluate(x_entrenamiento[val_index], y_entrenamiento[val_index])
        # _, acierto_prueba = modelo_cnn.evaluate(x_prueba, y_prueba)

        print("Acierto en el entrenamiento: {}%".format("%.2f" % (acierto_entrenamiento*100)))
        print("Acierto en la validación: {}%".format("%.2f" % (acierto_validacion*100)))
        # print("Acierto en la prueba: {}%".format("%.2f" % (acierto_prueba*100)))

        resultados_finales.append([acierto_entrenamiento,
                                    acierto_validacion])

        y_prediccion = modelo_cnn.predict(x_entrenamiento[val_index])
        razon_falsos_positivos = dict()
        razon_verdaderos_positivos = dict()
        area_bajo_curva_roc = dict()
        for j in range(cantidad_clases):
            razon_falsos_positivos[j], razon_verdaderos_positivos[j], _ = roc_curve(y_entrenamiento[val_index][:, j], y_prediccion[:, j])
            area_bajo_curva_roc[j] = auc(razon_falsos_positivos[j], razon_verdaderos_positivos[j])
        areas_roc.append(area_bajo_curva_roc)
    # Fin del for de repeticiones

    resultados_finales = np.asarray(resultados_finales)

    # Medias
    promedio_acierto_entrenamiento = statistics.mean(resultados_finales[:, 0])
    promedio_acierto_validacion = statistics.mean(resultados_finales[:, 1])
    # promedio_acierto_prueba = statistics.mean(resultados_finales[:, 2])

    # Desvíos estándar
    desvio_acierto_entrenamiento = statistics.stdev(resultados_finales[:, 0])
    desvio_acierto_validacion = statistics.stdev(resultados_finales[:, 1])
    # desvio_acierto_prueba = statistics.stdev(resultados_finales[:, 2])

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
    # print("\tTop de palabras frecuentes utilizadas: {}".format(top_palabras_frecuentes))
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

    # print("\tAcierto en el prueba [media, desvío]: [{}% / {}%]".format("%.2f" % (promedio_acierto_prueba*100), "%.2f" % (100*desvio_acierto_prueba)))
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