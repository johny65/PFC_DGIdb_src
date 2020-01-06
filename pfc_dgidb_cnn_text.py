# Paquetes
from __future__ import absolute_import, division, print_function, unicode_literals # Compatibilidad entre Python 2 y 3

from redes_neuronales_preprocesamiento import cargar_ejemplos # Carga de ejemplos
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, Conv2D, MaxPooling2D # Conv1D, MaxPooling1D,
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import funciones_auxiliares as fa

from sklearn.model_selection import StratifiedKFold,KFold

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

x_entrenamiento, y_entrenamiento = cargar_ejemplos(etiquetas_neural_networks_ruta,
                                                    ejemplos_directorio,
                                                    out_interacciones_ruta,
                                                    embeddings_file=embeddings_file,
                                                    top_palabras=top_palabras_frecuentes,
                                                    max_longitud=maxima_longitud_ejemplos,
                                                    incluir_sin_interacciones=True)

k = 5
folds_dict = fa.kfolding(k,x_entrenamiento)

print("Cantidad de ejemplos: {}".format(x_entrenamiento.shape[0]))
print("Dimension del embedding: {}".format(x_entrenamiento.shape[1]))
print("Longitud de los ejemplos: {}".format(x_entrenamiento.shape[2]))

cantidad_ejemplos = x_entrenamiento.shape[0]
filas_dimension_embedding = x_entrenamiento.shape[1]
columnas_maxima_longitud_ejemplos = x_entrenamiento.shape[2]
canal_color_escala_grises = 1

x_entrenamiento = x_entrenamiento.reshape(cantidad_ejemplos,filas_dimension_embedding,columnas_maxima_longitud_ejemplos,canal_color_escala_grises)

''' Variables globales '''
PORCENTAJE_DROPEO = 0.3 # Pone en 0 el #% de los datos aleatoriamente
CANTIDAD_FILTROS = 16 # Cantidad de filtro de convolución
DIMENSION_KERNEL = 3 # De 1x3
DIMENSION_POOLING = 2
NEURONAS_OCULTAS = 30 # 96
VELOCIDAD_APRENDIZAJE = 1e-3
CANTIDAD_EPOCAS = 5
PORCENTAJE_VALIDACION = 0.2 # 20% de ejemplos usados para validar. Son tomados desde el final
DIMENSION_BACHA = 512

for i in range(0,k,1):
    print("Particion: {}/{}".format(i+1,k))

    # Red neuronal convolucional
    modelo_cnn = Sequential() # Modelo secuencial (una capa detrás de la otra). Perceptrón multicapa (Arquitectura)

    '''
    Una capa de convolución seguida de una capa de polling tantas veces como se desee.
    Esa es la característica principal de una CNN.
    '''

    formato_entrada = (filas_dimension_embedding,columnas_maxima_longitud_ejemplos,canal_color_escala_grises) # ancho,alto,canales

    modelo_cnn.add(Conv2D(filters=CANTIDAD_FILTROS,
                        kernel_size=(1,DIMENSION_KERNEL),
                        activation='relu',
                        input_shape=formato_entrada))
    modelo_cnn.add(MaxPooling2D(pool_size=(1,DIMENSION_POOLING)))

    formato_entrada = (filas_dimension_embedding,
                (columnas_maxima_longitud_ejemplos-DIMENSION_KERNEL+1)/DIMENSION_POOLING,
                canal_color_escala_grises)

    modelo_cnn.add(Conv2D(filters=CANTIDAD_FILTROS,
                    kernel_size=(1,DIMENSION_KERNEL),
                    activation='relu',
                    input_shape=formato_entrada))
    modelo_cnn.add(MaxPooling2D(pool_size=(1,DIMENSION_POOLING)))

    modelo_cnn.add(Flatten()) # input_shape=(imagen_ancho,imagen_alto)

    # Capa de entrada
    modelo_cnn.add(Dropout(PORCENTAJE_DROPEO)) # Dropout: se utiliza para evitar el sobreentrenamiento
    modelo_cnn.add(Dense(NEURONAS_OCULTAS,activation='relu')) # Siempre relu en las capas ocultas

    # Capa de salida
    modelo_cnn.add(Dropout(PORCENTAJE_DROPEO)) # Dropout: se utiliza para evitar el sobreentrenamiento
    modelo_cnn.add(Dense(20,activation='softmax')) # Softmax porque hay más de dos clases

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
    optimizador = Adam(lr=VELOCIDAD_APRENDIZAJE)
    modelo_cnn.compile(optimizer='adam', # Velocidad de aprendizaje optimizer=optimizador
                loss='categorical_crossentropy', # Función de error. Categorical_crossentropy porque hay más de dos clases
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
    # callbacks = [EarlyStopping(monitor='val_accuracy',patience=1)] # 'val_loss'

    # Detalles del modelo neuronal
    modelo_cnn.summary()

    # Entrenamiento del modelo
    registro = modelo_cnn.fit(x_entrenamiento[folds_dict[i][0]],
                            y_entrenamiento[folds_dict[i][0]],
                            epochs=CANTIDAD_EPOCAS,
                            # callbacks=callbacks,
                            # validation_split=PORCENTAJE_VALIDACION,
                            validation_data=(x_entrenamiento[folds_dict[i][1]],y_entrenamiento[folds_dict[i][1]]),
                            verbose=2) # Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch
                            # batch_size=DIMENSION_BACHA)

# fa.graficas(registro)

# Evaluación del modelo
# prueba_error,prueba_acierto = model.evaluate(x_prueba,y_prueba)

# Hacer predicciones
# predicciones = model.predict(x_prueba)