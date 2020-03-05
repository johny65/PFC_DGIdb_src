import csv
import numpy as np
import pandas as pd
import pathlib
import pickle
import random

from datos_preprocesamiento import cargar_publicaciones
from redes_neuronales_preprocesamiento import cargar_interacciones, armar_test_set
from funciones_auxiliares import evaluar

from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, GlobalMaxPool1D, Dropout, Conv1D, Activation, LSTM, SpatialDropout1D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight


EJEMPLOS_DIRECTORIO = "replaced4"
INTERACCIONES_RUTA = "interacciones_lista.txt"
ETIQUETAS_RUTA = "etiquetas_neural_networks_4_v2.csv"
TOP_PALABRAS = 1000
MAXIMA_LONGITUD_EJEMPLOS = 500
LONGITUD_PALABRAS = 4
ARCHIVO_X = "xs/x4_{}x{}.pickle".format(MAXIMA_LONGITUD_EJEMPLOS, LONGITUD_PALABRAS)
INCLUIR_SIN_INTERACCION = False
MODELO_SIMPLE = "model-simple.h5"
MODELO_CONV = "model-conv.h5"
# MODELO_RNN = "rnn/rnnv2-16clases-500x4-100ue-01do"
MODELO_RNN = "rnn/rnn-16clases-500x4-100-masépocas"
EPOCAS = 100
UNIDADES_LSTM = 100


def get_tokenizer():
    global contenido_dict
    tokenizer_file = pathlib.Path("tokenizer4.pickle")
    if tokenizer_file.exists():
        with open(tokenizer_file, "rb") as f:
            vocabulario = pickle.load(f)
    else:
        vocabulario = Tokenizer(lower=False)
        contenido_dict = cargar_publicaciones(EJEMPLOS_DIRECTORIO)
        vocabulario.fit_on_texts(contenido_dict.values())
        with open(tokenizer_file, "wb") as f:
            pickle.dump(vocabulario, f)
    print("Listo fit on texts.")
    return vocabulario

vocabulario = get_tokenizer()
gen_mark = "<GEN>"; gen_mark_index = max(vocabulario.index_word) + 1
droga_mark = "<DROGA>"; droga_mark_index = gen_mark_index + 1



def cargar_ejemplos(etiquetas_neural_networks_ruta, ejemplos_directorio, interacciones,
                    porcentaje_test=0.0, incluir_sin_interaccion=True, shuffle=True):
    """Carga ejemplos."""

    with open(etiquetas_neural_networks_ruta, encoding="utf8") as enn_csv:
        lector_csv = csv.reader(enn_csv, delimiter=',', quoting=csv.QUOTE_ALL)
        if incluir_sin_interaccion:
            etiquetas = [row for row in lector_csv]
        else:
            etiquetas = [row for row in lector_csv if row[3] != "sin_interaccion"]
    total = len(etiquetas)

    x_file = pathlib.Path(ARCHIVO_X)
    if x_file.exists():
        with open(x_file, "rb") as f:
            x = pickle.load(f)
    
    else:
        contenido_dict = cargar_publicaciones(ejemplos_directorio) # tiene un elemento por artículo: pmid -> contenido

        # marcadores específicos para gen/droga de interés:
        gen_mark = "<GEN>"; gen_mark_index = max(vocabulario.index_word) + 1
        droga_mark = "<DROGA>"; droga_mark_index = gen_mark_index + 1
        print("GMI:", gen_mark_index)

        vocabulario.word_index[gen_mark] = gen_mark_index
        vocabulario.index_word[gen_mark_index] = gen_mark
        vocabulario.word_index[droga_mark] = droga_mark_index
        vocabulario.index_word[droga_mark_index] = droga_mark

        items = list(contenido_dict.items())
        sequences = vocabulario.texts_to_sequences((i[1] for i in items))
        sequences = {items[i][0]: sequences[i] for i in range(len(items))}
        print("Listo texts to sequences.")

        def add(lista, e):
            """Agrega el elemento e a la colección sin que se repita consecutivamente."""
            if not lista or (lista and lista[-1] != e):
                lista.append(e)

        x = []
        for i in range(total):
            print("Cargando ejemplo {}/{}".format(i+1, total))
            pmid = etiquetas[i][0]
            ejemplo = sequences[pmid]
            gen = etiquetas[i][1]
            droga = etiquetas[i][2]

            num_words = TOP_PALABRAS
            ready = False
            while not ready:
                ejemplo_reducido = []
                for elemento in ejemplo:
                    if elemento == gen_mark_index or elemento == droga_mark_index:
                        add(ejemplo_reducido, elemento)
                    else:
                        palabra = vocabulario.index_word[elemento]
                        if palabra.startswith("xxx"):
                            # gen o droga de interés
                            if palabra == "xxx{}xxx".format(gen):
                                de_interes = gen_mark_index
                            elif palabra == "xxx{}xxx".format(droga):
                                de_interes = droga_mark_index
                            else:
                                de_interes = None
                            if de_interes:
                                add(ejemplo_reducido, de_interes)
                        elif elemento <= num_words and len(palabra) >= LONGITUD_PALABRAS:
                            # agregar sólo si está dentro de máxima longitud
                            add(ejemplo_reducido, elemento)
                if len(ejemplo_reducido) <= MAXIMA_LONGITUD_EJEMPLOS or num_words < 0:
                    ready = True
                else:
                    num_words -= 1
                    ejemplo = ejemplo_reducido
            
            x.append(ejemplo_reducido)
        
        x = pad_sequences(x, maxlen=MAXIMA_LONGITUD_EJEMPLOS)
        with open(x_file, "wb") as f:
            pickle.dump(x, f)

    # carga de y
    y = []
    for i in range(total):
        interaccion = etiquetas[i][3]
        y.append(interaccion if interaccion in interacciones else "other")
    
    if shuffle:
        seed = random.random()
        random.seed(seed)
        random.shuffle(x)
        random.seed(seed)
        random.shuffle(y)

    return x, y


def entrenar(entrenar_simple, entrenar_conv, entrenar_rnn, modelo=None):
    interacciones = cargar_interacciones(INTERACCIONES_RUTA, invertir=True)
    x, y = cargar_ejemplos(ETIQUETAS_RUTA, EJEMPLOS_DIRECTORIO, interacciones,
                           incluir_sin_interaccion=INCLUIR_SIN_INTERACCION, shuffle=False)
    
    yun = np.unique(y)
    pesos_clases = compute_class_weight("balanced", yun, y)
    print("Pesos de clases ({}): {}".format(len(pesos_clases), list(zip(yun, pesos_clases))))

    y = [[e] for e in y]
    mlb = MultiLabelBinarizer()
    mlb.fit(y)
    interacciones = mlb.classes_
    print("Interacciones:", len(interacciones))
    y = mlb.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9000)

    # red
    distinct_words = set()
    for e in x:
        distinct_words.update(e)
    max_words = len(distinct_words)
    print("Cantidad de palabras:", max_words)
    # distinct_words = sorted(distinct_words)
    # distinct_words[
    max_words = droga_mark_index + 1
    print("Cantidad de palabras:", max_words)
    num_classes = len(interacciones)
    print("Cantidad de clases:", num_classes)
    
    
    if entrenar_rnn:
        modelo = MODELO_RNN
        model = Sequential()
        # model.add(Embedding(input_dim=max_words, output_dim=UNIDADES_LSTM, input_length=MAXIMA_LONGITUD_EJEMPLOS))
        # model.add(SpatialDropout1D(0.1))
        # model.add(LSTM(units=UNIDADES_LSTM, dropout=0.1, recurrent_dropout=0.1))
        # model.add(Dense(UNIDADES_LSTM, activation='relu'))
        # model.add(Dropout(0.1))
        # model.add(Dense(num_classes, activation='softmax'))
    
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        model.add(Embedding(input_dim=max_words, output_dim=UNIDADES_LSTM, input_length=MAXIMA_LONGITUD_EJEMPLOS))
        # model.add(Dropout(0.1))
        model.add(LSTM(units=UNIDADES_LSTM, dropout=0.1)) #, input_shape=formato_entrada))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        model.summary()

        callbacks = [
            ReduceLROnPlateau(),
            EarlyStopping(patience=4),
            ModelCheckpoint(filepath=modelo, save_best_only=True)
        ]

        history = model.fit(x_train, y_train,
                            class_weight=pesos_clases,
                            epochs=EPOCAS,
                            batch_size=32,
                            validation_split=0.1,
                            callbacks=callbacks)

        with open(MODELO_RNN + ".history", "wb") as f:
            pickle.dump(history, f)

    loaded_model = load_model(modelo)
    acc, pred = evaluar(loaded_model, x_test, y_test)
    print("Acc:", acc)



if __name__ == "__main__":
    entrenar(False, False, True, "model-rnn.h5")
