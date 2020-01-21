# Paquetes
from __future__ import absolute_import, division, print_function, unicode_literals # Compatibilidad entre Python 2 y 3
import numpy as np
from redes_neuronales_preprocesamiento import cargar_ejemplos # Carga de ejemplos
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Concatenate, Dropout, Flatten, Dense, concatenate, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam,SGD # SGD: Gradiente descendiente
from keras.utils import plot_model
import funciones_auxiliares as fa
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Para utilizar CPU en lugar de GPU

''' Carga de datos '''
etiquetas_neural_networks_ruta = "etiquetas_neural_networks2.csv"
ejemplos_directorio = "replaced_old"
out_interacciones_ruta = "interacciones_lista.txt"
embeddings_file = "glove.6B.50d.txt"
# embeddings_file = "glove.6B.100d.txt"
# embeddings_file = "glove.6B.200d.txt"
# embeddings_file = "glove.6B.300d.txt"
top_palabras_frecuentes = 100
maxima_longitud_ejemplos = 500
longitud_palabras_mayor_a = 3
cantidad_ejemplos_sin_interaccion = 3144 # old: 3144; new: 2944

# maxima_longitud_ejemplos: 300 -> top_palabras_frecuentes: 64
# maxima_longitud_ejemplos: 500 -> top_palabras_frecuentes: 103
# maxima_longitud_ejemplos: 1000 -> top_palabras_frecuentes: 552
# maxima_longitud_ejemplos: 1500 -> top_palabras_frecuentes: 790
# maxima_longitud_ejemplos: 2000 -> top_palabras_frecuentes: 1112
# maxima_longitud_ejemplos: 2500 -> top_palabras_frecuentes: 1298
# maxima_longitud_ejemplos: 3000 -> top_palabras_frecuentes: 1421

''' Parámetros del modelo '''
k = 10 # Número de particiones para la validación cruzada
PORCENTAJE_DROPEO = 0.4 # Pone en 0 el #% de los datos aleatoriamente
CANTIDAD_FILTROS = 10 # Cantidad de filtro de convolución
DIMENSION_KERNEL = (3, 5)
CANTIDAD_EPOCAS = 15
PORCENTAJE_VALIDACION = 0.2
PORCENTAJE_PRUEBA = 0.2
DIMENSION_BACHA = 32
ACTIVACION_OCULTA = 'relu'
ACTIVACION_SALIDA = 'softmax'
VELOCIDAD_APRENDIZAJE = 1e-3
OPTIMIZADOR = "adam" # 'Adam(lr=VELOCIDAD_APRENDIZAJE)'
FUNCION_ERROR = 'categorical_crossentropy'
CALLBACK_PARADA = True
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
                                                           porcentaje_test=PORCENTAJE_PRUEBA)

cantidad_ejemplos = x_entrenamiento.shape[0]
cantidad_ejemplos_entrenamiento = 0
cantidad_ejemplos_validacion = 0
cantidad_ejemplos_prueba = x_prueba.shape[0]
dimension_embedding = x_entrenamiento.shape[2] # Columnas
cantidad_clases = y_entrenamiento.shape[1]
NEURONAS_SALIDA = cantidad_clases
NEURONAS_OCULTAS = NEURONAS_SALIDA*3

if MODELO_FINAL: # Entrenar modelo final
    print("Entrenando modelo final...")
    x_entrenamiento = np.concatenate((x_entrenamiento, x_prueba))
    y_entrenamiento = np.concatenate((y_entrenamiento, y_prueba))

    modelo_cnn = Sequential()
    formato_entrada = (maxima_longitud_ejemplos, dimension_embedding)

    '''Intento de entrenar con kernels de distintos tamaños al mismo tiempo'''
    submodelos = list()
    for i in range(0, len(DIMENSION_KERNEL), 1):
        submodelos.append(Sequential())
        submodelos[i].add(Conv1D(input_shape=formato_entrada,
                          filters=CANTIDAD_FILTROS,
                          kernel_size=DIMENSION_KERNEL[i],
                          padding='same',
                          bias_initializer='random_uniform'))
        submodelos[i].add(MaxPooling1D(pool_size=maxima_longitud_ejemplos))
    convoluciones_poolings = Concatenate(submodelos, input_shape=(len(DIMENSION_KERNEL), 1))
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
    
    # modelo_cnn.add(Dropout(PORCENTAJE_DROPEO))
    modelo_cnn.add(Flatten())

    modelo_cnn.add(Dropout(PORCENTAJE_DROPEO))
    modelo_cnn.add(Dense(NEURONAS_OCULTAS, activation = ACTIVACION_OCULTA))

    modelo_cnn.add(Dropout(PORCENTAJE_DROPEO))
    modelo_cnn.add(Dense(NEURONAS_SALIDA, activation = ACTIVACION_SALIDA))

    modelo_cnn.compile(optimizer = OPTIMIZADOR,
                    loss = FUNCION_ERROR,
                    metrics = ['accuracy'])

    if CALLBACK_PARADA:
        parada_temprana = [EarlyStopping(monitor = 'loss',
                                        patience = 0,
                                        verbose = 2,
                                        mode = 'auto')]
    else:
        parada_temprana = None
                                    
    modelo_cnn.summary()

    registro = modelo_cnn.fit(x=x_entrenamiento,
                              y=y_entrenamiento,
                              epochs=CANTIDAD_EPOCAS,
                              callbacks=parada_temprana,
                              verbose=1,
                              batch_size=DIMENSION_BACHA,
                              shuffle=True)

    # Guardado del modelo
    modelo_ruta = "cnn.h5"
    modelo_cnn.save(modelo_ruta)
    print("Modelo final entrenado y guardado.")
else: # Análisis del modelo
    folds_dict = fa.kfolding(k, cantidad_ejemplos, PORCENTAJE_VALIDACION) # Se obtienen las particiones para realizar la validación cruzada

    suma_acierto_entrenamiento = 0
    suma_acierto_validacion = 0
    suma_acierto_prueba = 0

    resultados_finales = list()

    for i in range(0,k,1):
        cantidad_ejemplos_entrenamiento = len(x_entrenamiento[folds_dict[i][0]])
        cantidad_ejemplos_validacion = len(x_entrenamiento[folds_dict[i][1]])

        ''' Modelo multikernel '''
        formato_entrada = (maxima_longitud_ejemplos, dimension_embedding,)
        entrada = Input(formato_entrada)
        capas = list()
        for j in range(0, len(DIMENSION_KERNEL), 1):
            convolucion = Conv1D(filters=CANTIDAD_FILTROS,
                                 kernel_size=DIMENSION_KERNEL[j],
                                 padding='same',
                                 bias_initializer='random_uniform')(entrada)
            pooling = MaxPooling1D(pool_size=maxima_longitud_ejemplos)(convolucion)
            capas.append(pooling)
        convoluciones_poolings = concatenate(capas)

        # dropout1 = Dropout(PORCENTAJE_DROPEO)(convoluciones_poolings)
        flatten = Flatten()(convoluciones_poolings)
        dropout2 = Dropout(PORCENTAJE_DROPEO)(flatten)
        # dense1 = Dense(NEURONAS_OCULTAS, activation = ACTIVACION_OCULTA)(dropout2)
        # dropout3 = Dropout(PORCENTAJE_DROPEO)(dense1)
        dense2 = Dense(NEURONAS_SALIDA, activation = ACTIVACION_SALIDA)(dropout2)
        modelo_cnn = Model(input=entrada, output=dense2)
        ''' '''

        # ''' Modelo monokernel '''
        # modelo_cnn = Sequential()
        # formato_entrada = (maxima_longitud_ejemplos, dimension_embedding)

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
        
        # # modelo_cnn.add(Dropout(PORCENTAJE_DROPEO))
        # modelo_cnn.add(Flatten())

        # modelo_cnn.add(Dropout(PORCENTAJE_DROPEO))
        # modelo_cnn.add(Dense(NEURONAS_OCULTAS, activation=ACTIVACION_OCULTA))

        # modelo_cnn.add(Dropout(PORCENTAJE_DROPEO))
        # modelo_cnn.add(Dense(NEURONAS_SALIDA, activation=ACTIVACION_SALIDA))
        # ''' '''

        modelo_cnn.compile(optimizer=OPTIMIZADOR, # Adam(lr=1e-6)
                           loss=FUNCION_ERROR,
                           metrics=['accuracy'])

        bajar_velocidad = ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.1,
                                            patience=10,
                                            verbose=0,
                                            mode='auto',
                                            min_delta=0.0001,
                                            cooldown=0,
                                            min_lr=0)

        if CALLBACK_PARADA:
            parada_temprana = EarlyStopping(monitor='val_loss',
                                            patience=0,
                                            verbose=2,
                                            mode='auto')
        else:
            parada_temprana = None
                                        
        modelo_cnn.summary()

        plot_model(modelo_cnn, to_file="modelo_cnn_{}_arquitectura.png".format(i+1))

        print("Particion: {}/{}".format(i+1,k))

        registro = modelo_cnn.fit(x=x_entrenamiento[folds_dict[i][0]],
                                  y=y_entrenamiento[folds_dict[i][0]],
                                  epochs = CANTIDAD_EPOCAS,
                                  callbacks = [parada_temprana, bajar_velocidad],
                                  validation_data = (x_entrenamiento[folds_dict[i][1]], y_entrenamiento[folds_dict[i][1]]),
                                  verbose = 1,
                                  batch_size = DIMENSION_BACHA)

        _, acierto_prueba = modelo_cnn.evaluate(x_prueba, y_prueba)

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
    print("\tCantidad de ejemplos cargados: {}".format(2944 + cantidad_ejemplos_sin_interaccion))
    print("\tCantidad de ejemplos con la etiqueta sin_interacción: {}".format(cantidad_ejemplos_sin_interaccion))
    print("\tCantidad de ejemplos utilizados para entrenar: {}".format(cantidad_ejemplos_entrenamiento))
    print("\tCantidad de ejemplos utilizados para validar: {}".format(cantidad_ejemplos_validacion))
    print("\tCantidad de ejemplos utilizados para probar: {}".format(cantidad_ejemplos_prueba))
    print("\tTop de palabras frecuentes utilizadas: {}".format(top_palabras_frecuentes))
    print("\tSolo palabras con longitud mayor a: {}".format(longitud_palabras_mayor_a))
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
    print("\tVelocidad de aprendizaje: {}".format(VELOCIDAD_APRENDIZAJE))
    print("\tOptimizador: {}".format(OPTIMIZADOR))
    print("\tLoss function: {}".format(FUNCION_ERROR))
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
    print("\tAcierto en la prueba promedio: {}".format(promedio_acierto_prueba))
    for i in range(0, len(resultados_finales), 1):
        print("\t\tAcierto en la prueba en la partición {}: {}".format(i+1, resultados_finales[i][2]))

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