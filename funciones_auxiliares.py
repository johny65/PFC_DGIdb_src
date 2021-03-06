from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from collections import Counter
import csv
import statistics
import random
import math
import os


def cargar_interacciones(in_file, invertir=False):
    """
    Carga el archivo de interacciones existentes de salida, y devuelve un diccionario
    con el índice de cada una para mapear el vector de salida. Por ejemplo:

        inhibitor
        agonist
        potentiator

    Cuando el caso de ejemplo sea de "inhibitor", el vector de salida será [1, 0, 0, 0].
    El último elemento se usa para agrupar las que no se nombran. Por ejemplo, con el listado
    anterior, cuando el caso de ejemplo sea de "modulator" el vector de salida será
    [0, 0, 0, 1].

    """
    with open(in_file, encoding="utf8") as f:
        if invertir:
            res = [l.strip() for l in f.readlines()]
        else:
            res = {l.strip(): i for i, l in enumerate(f.readlines())}
    # res["other"] = len(res)
    return res

def histograma(secuencias, secuencia_gen, secuencia_droga):
    tot = 32000
    posiciones_genes = []
    posiciones_drogas = []
    for ejemplo in secuencias:
        for p, elemento in enumerate(ejemplo):
            if elemento == secuencia_gen:
                if p < tot:
                    posiciones_genes.append(p)
            if elemento == secuencia_droga:
                if p < tot:
                    posiciones_drogas.append(p)
    plt.figure()
    plt.hist(posiciones_genes, bins=150)
    plt.show()
    plt.figure()
    plt.hist(posiciones_drogas, bins=150)
    plt.show()

def mostrar_imagen(imagen,etiqueta):
    plt.figure()
    plt.imshow(imagen)
    plt.xlabel(etiqueta)
    plt.colorbar()
    plt.grid(False)
    plt.show()
    
def mostrar_cuadricula(x_entrenamiento,y_entrenamiento):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_entrenamiento[i], cmap=plt.cm.binary)
        plt.xlabel(y_entrenamiento[i])
    plt.show()

def graficas(registro_entrenamiento):
    etiquetas_entrenamiento = registro_entrenamiento.history
    
    acierto_entrenamiento = etiquetas_entrenamiento['accuracy']
    acierto_validacion = etiquetas_entrenamiento['val_accuracy']
    error_entrenamiento = etiquetas_entrenamiento['loss']
    error_validacion = etiquetas_entrenamiento['val_loss']    
        
    epocas = range(1,len(acierto_entrenamiento)+1)
    
    plt.plot(epocas,error_entrenamiento,'r',label='Error en entrenamiento')
    plt.plot(epocas,error_validacion,'b',label='Error en validación')
    plt.title('Errores de entrenamiento y validación')
    plt.xlabel('Épocas')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
    
    plt.clf()   # clear figure
    
    plt.plot(epocas,acierto_entrenamiento,'r',label='Acierto en el entrenamiento')
    plt.plot(epocas,acierto_validacion,'b',label='Acierto en la validación')
    plt.title('Acierto en el entrenamiento y la validación')
    plt.xlabel('Epocas')
    plt.ylabel('Tasa de acierto')
    plt.legend()
    
    plt.show()

def kfolding(particiones, cantidad_ejemplos, porcentaje_validacion):
    '''
    Retorna un diccionario donde la clave es el número de la partición y el valor una lista de dos elementos.
    El primer elemento de esta lista son los índices para el conjunto de entrenamiento.
    El segundo elemento de esta lista son los índices para el conjunto de validación.
    '''
    folds_dict = dict()
    cantidad_ejemplos_validacion = int(porcentaje_validacion*cantidad_ejemplos)
    paso = 1/(particiones*porcentaje_validacion)
    indices_entrenamiento = np.arange(cantidad_ejemplos)
    indices_extra = np.arange(int(cantidad_ejemplos_validacion - (cantidad_ejemplos_validacion*paso)))
    indices = np.concatenate((indices_entrenamiento, indices_extra))
    for k in range(0, particiones, 1):
        inicio = int(paso*k*cantidad_ejemplos_validacion)
        fin = int(inicio + cantidad_ejemplos_validacion)
        indices_validacion = indices[inicio:fin]
        indices_entrenamiento = list()
        for i in range(0, cantidad_ejemplos, 1):
            if i not in indices_validacion:
                indices_entrenamiento.append(i)
        indices_entrenamiento = np.asarray(indices_entrenamiento)
        folds_dict[k] = [indices_entrenamiento, indices_validacion]
    return folds_dict

def graficar_curva_roc(cantidad_clases, y_prueba, y_prediccion):
    '''
    razon_falsos_positivos: eje X en la cuerva ROC.
    razon_verdaderos_positivos: eje Y en la cuerva ROC.
    area_bajo_curva_roc: [0 - 1]
    '''
    razon_falsos_positivos = dict()
    razon_verdaderos_positivos = dict()
    area_bajo_curva_roc = dict()
    plt.figure()
    ancho_linea = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=ancho_linea, linestyle='--')
    for i in range(cantidad_clases):
        razon_falsos_positivos[i], razon_verdaderos_positivos[i], _ = roc_curve(y_prueba[:, i], y_prediccion[:, i])
        area_bajo_curva_roc[i] = auc(razon_falsos_positivos[i], razon_verdaderos_positivos[i])
        plt.plot(razon_falsos_positivos[i], razon_verdaderos_positivos[i], label='Clase {}: {}'.format(i, area_bajo_curva_roc[i]), lw=ancho_linea)
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.title('Curvas ROC')
        plt.xlabel('Razón de falsos positivos')
        plt.ylabel('Razón de verdaderos positivos')
        plt.legend()
        plt.show()
        plt.clf()

def balancear_clases(etiquetas_archivo_ruta, # Archivo de etiquetas: pmid, gen, droga, interacción
                     interacciones_lista_ruta, # Lista de etiquetas a considerar
                     excluir_interacciones_lista, # Lista de interacciones que no se cargarán
                     ejemplos_cantidad, # Cantidad de ejemplos a cargar
                     porcentaje_prueba,
                     balancear): # Porcentaje de los ejemplos que se utilizarán para la prueba
    '''
    Devuelve (training, test): los conjuntos de interacciones fármaco-gen para
    entrenamiento y prueba balanceados en cantidad de ejemplos por clase.
    Se recomienda que al menos haya dos ejemplos por clase.
    '''
    ifg_balanceadas_prueba_lista = list()
    ifg_balanceadas_entrenamiento_lista = list()

    # Carga de las interacciones a considerar en una lista
    interacciones_considerar = list()
    with open(interacciones_lista_ruta, encoding="utf8") as interacciones:
        for interaccion in interacciones:
            interacciones_considerar.append(interaccion.strip())

    # Carga de las interacciones fármaco-gen
    pmids_lista = list()
    genes_lista = list()
    drogas_lista = list()
    interacciones_lista = list()
    with open(etiquetas_archivo_ruta, encoding="utf8") as etiquetas:
        lector_csv = csv.reader(etiquetas, delimiter=',', quoting=csv.QUOTE_ALL)
        for linea in lector_csv:
            if linea[3] not in excluir_interacciones_lista:
                pmids_lista.append(linea[0])
                genes_lista.append(linea[1])
                drogas_lista.append(linea[2])
                if linea[3] not in interacciones_considerar:
                    interacciones_lista.append("other")
                else:
                    interacciones_lista.append(linea[3])

    # Armado de los conjuntos balanceados de entrenamiento y prueba
    interacciones_cantidad_dict = Counter(interacciones_lista) # Clases y cantidad de ejemplos por clases ordenados de mayor a menor
    cantidad_clases = len(interacciones_cantidad_dict)

    if porcentaje_prueba == 0.0: # Para el caso donde no se desee un conjunto de prueba
        cantidad_ejemplos_prueba_por_clase = 0
        cantidad_minima_ejemplos = 0
    else:
        cantidad_minima_ejemplos = math.ceil(cantidad_clases/porcentaje_prueba)

    if ejemplos_cantidad < cantidad_minima_ejemplos: # Ajusta al número de ejemplos mínimo necesario para el armado de los conjuntos en caso de ser necesario
        ejemplos_cantidad = cantidad_minima_ejemplos

    if balancear: # Cargar balanceando
        cantidad_ejemplos_prueba_por_clase = int((ejemplos_cantidad*porcentaje_prueba)/cantidad_clases)
        cantidad_ejemplos_entrenamiento_por_clase = int((ejemplos_cantidad-(cantidad_ejemplos_prueba_por_clase*cantidad_clases))/cantidad_clases)
        ejemplos_por_clase = int(ejemplos_cantidad/cantidad_clases)
        seed = random.random()
        random.seed(seed)
        for interaccion, cantidad in interacciones_cantidad_dict.items():
            ifg_interaccion_lista = list()
            for i in range(0, len(pmids_lista), 1): # Agrega a "ifg_interaccion_lista" las ifg con la interacción "interaccion"
                if interacciones_lista[i] == interaccion:
                    ifg_interaccion_lista.append([pmids_lista[i], genes_lista[i], drogas_lista[i], interacciones_lista[i]])
            if cantidad >= ejemplos_por_clase:
                for ifg in random.sample(ifg_interaccion_lista, k=cantidad_ejemplos_prueba_por_clase):
                    ifg_balanceadas_prueba_lista.append(ifg)
                    ifg_interaccion_lista.remove(ifg)
                for ifg in random.sample(ifg_interaccion_lista, k=cantidad_ejemplos_entrenamiento_por_clase):
                    ifg_balanceadas_entrenamiento_lista.append(ifg)
            else:
                cantidad_ejemplos_prueba_clase_actual = math.ceil(cantidad*porcentaje_prueba)
                ifg_interaccion_prueba_lista = random.sample(ifg_interaccion_lista, k=cantidad_ejemplos_prueba_clase_actual)
                ifg_interaccion_entrenamiento_lista = list()
                for ifg in ifg_interaccion_lista:
                    if ifg not in ifg_interaccion_prueba_lista:
                        ifg_interaccion_entrenamiento_lista.append(ifg)
                for ifg in random.choices(ifg_interaccion_prueba_lista, k=cantidad_ejemplos_prueba_por_clase):
                    ifg_balanceadas_prueba_lista.append(ifg)
                if ifg_interaccion_entrenamiento_lista == []: # Para cuando solo hay un ejemplo y queda en el conjunto de prueba
                    break
                for ifg in random.choices(ifg_interaccion_entrenamiento_lista, k=cantidad_ejemplos_entrenamiento_por_clase):
                    ifg_balanceadas_entrenamiento_lista.append(ifg)
    else: # Cargar sin balancear
        for interaccion, cantidad in interacciones_cantidad_dict.items():
            ifg_interaccion_lista = list()
            for i in range(0, len(pmids_lista), 1): # Agrega a "ifg_interaccion_lista" las ifg con la interacción "interaccion"
                if interacciones_lista[i] == interaccion:
                    ifg_interaccion_lista.append([pmids_lista[i], genes_lista[i], drogas_lista[i], interacciones_lista[i]])
            cantidad_ejemplos_prueba_clase_actual = math.ceil(cantidad*porcentaje_prueba)
            for ifg in random.sample(ifg_interaccion_lista, k=cantidad_ejemplos_prueba_clase_actual):
                ifg_balanceadas_prueba_lista.append(ifg)
                ifg_interaccion_lista.remove(ifg)
            for ifg in ifg_interaccion_lista:
                ifg_balanceadas_entrenamiento_lista.append(ifg)

    return ifg_balanceadas_entrenamiento_lista, ifg_balanceadas_prueba_lista

def generar_pesos_clases(etiquetas_neural_networks_ruta,
                         excluir_interacciones_lista,
                         interacciones_considerar):
    interaccion_por_etiqueta = list()
    with open(etiquetas_neural_networks_ruta, encoding="utf8") as etiquetas:
        lector_csv = csv.reader(etiquetas, delimiter=',', quoting=csv.QUOTE_ALL)
        for linea in lector_csv:
            if linea[3] not in excluir_interacciones_lista:
                if linea[3] not in interacciones_considerar:
                    interaccion_por_etiqueta.append("other")
                else:
                    interaccion_por_etiqueta.append(linea[3])
    interacciones_cantidad_dict = Counter(interaccion_por_etiqueta) # Clases y cantidad de ejemplos por clases ordenados de mayor a menor
    cantidad_elementos_mayor_clase = max(interacciones_cantidad_dict.values())
    interacciones_pesos_dict = dict()
    # numero_clase = 0
    # for cantidad in interacciones_cantidad_dict.values():
    #     interacciones_pesos_dict[numero_clase] = cantidad/cantidad_elementos_mayor_clase
    #     numero_clase += 1
    clases_unicas = np.unique(interaccion_por_etiqueta)
    pesos = compute_class_weight("balanced", clases_unicas, interaccion_por_etiqueta)
    pesos = dict(zip(clases_unicas, pesos))
    for i, interaccion in enumerate(interacciones_considerar):
        if interaccion not in excluir_interacciones_lista:
            interacciones_pesos_dict[i] = pesos[interaccion]
    # print(interacciones_pesos_dict)
    # Normalización de los pesos
    # mayor_peso = max(interacciones_pesos_dict.values())
    # for clase, peso in interacciones_pesos_dict.items():
    #     interacciones_pesos_dict[clase] = peso/mayor_peso

    return interacciones_pesos_dict


def evaluar(modelo, x, y_real):
    """Evalúa el modelo dado calculando la precisión teniendo en cuenta si acierta o no
    la clase predicha con la real.
    
    Devuelve el porcentaje de precisión y la salida predicha.
    """

    pred_orig = modelo.predict(x)
    pred = [np.asarray([1 if i == max(p1) else 0 for i in p1]) for p1 in pred_orig]

    if len(pred) != len(y_real):
        exit("La longitud de la salida deseada es distinta a la longitud de la salida de la red.")
    
    res = [any(p & y) for (p, y) in zip(pred, y_real)]
    acc = np.count_nonzero(res) / len(res) * 100.0
    return acc, pred_orig


def test_split(interacciones_ruta, etiquetas_ruta, porcentaje_test):
    """Separa un conjunto de test para siempre, a partir de las etiquetas."""
    interacciones = cargar_interacciones(interacciones_ruta, invertir=True)
    etiquetas = []
    clases = []
    with open(etiquetas_ruta, encoding="utf8") as f:
        for row in f:
            row = row.strip()
            clase = row.split(',')[3]
            clase = clase if clase in interacciones else "other"
            etiquetas.append(row)
            clases.append(clase)
    print("Cantidad de interacciones:", len(interacciones))
    print("Cantidad de clases de los ejemplos:", len(set(clases)))
    print("Cantidad de ejemplos:", len(etiquetas))

    x_train, x_test = train_test_split(etiquetas, test_size=porcentaje_test, stratify=clases)
    print("Cantidad de ejemplos para train:", len(x_train))
    print("Cantidad de ejemplos para test:", len(x_test))
    print(x_train[0], x_test[0])
    print("Total:", len(x_train) + len(x_test))
    
    with open("etiquetas_train.csv", "w", encoding="utf8") as out:
        for e in x_train:
            out.write(e + '\n')
    with open("etiquetas_test.csv", "w", encoding="utf8") as out:
        for e in x_test:
            out.write(e + '\n')


# ----------------------------------------------------------------------------------

def ifg_entrenamiento_prueba_sin_reejemplificacion(etiquetas_archivo_ruta, # Archivo de etiquetas: pmid, gen, droga, interacción
                                                    interacciones_lista_ruta, # Lista de etiquetas a considerar
                                                    excluir_interacciones_lista, # Lista de interacciones que no se cargarán
                                                    porcentaje_prueba, # Porcentaje de los ejemplos que se utilizarán para la prueba
                                                    cantidad_ejemplos_sin_interaccion): 
    '''
    Devuelve los conjuntos de interacciones fármaco-gen para entrenamiento y prueba.
    No se cargan ifg sin_interaccion en el conjunto de prueba.
    '''
    ifg_prueba_lista = list()
    ifg_entrenamiento_lista = list()

    # Carga de las interacciones a considerar en una lista
    interacciones_considerar = list()
    with open(interacciones_lista_ruta, encoding="utf8") as interacciones:
        for interaccion in interacciones:
            interacciones_considerar.append(interaccion.strip())

    # Carga de las interacciones fármaco-gen
    pmids_lista = list()
    genes_lista = list()
    drogas_lista = list()
    interacciones_lista = list()
    with open(etiquetas_archivo_ruta, encoding="utf8") as etiquetas:
        lector_csv = csv.reader(etiquetas, delimiter=',', quoting=csv.QUOTE_ALL)
        for linea in lector_csv:
            if linea[3] not in excluir_interacciones_lista:
                pmids_lista.append(linea[0])
                genes_lista.append(linea[1])
                drogas_lista.append(linea[2])
                if linea[3] not in interacciones_considerar:
                    interacciones_lista.append("other")
                else:
                    interacciones_lista.append(linea[3])

    # otra_lista = list()
    # for pmid_unico in dict.fromkeys(pmids_lista):
    #     genes_por_pmid = list()
    #     drogas_por_pmid = list()
    #     for i in range(0, len(pmids_lista), 1):
    #         if pmid_unico == pmids_lista[i] and interacciones_lista[i] != "sin_interaccion":
    #             genes_por_pmid.append(genes_lista[i])
    #             drogas_por_pmid.append(drogas_lista[i])
    #         interacciones_genes = list()
    #         interacciones_drogas = list()
    #         for 


    # pmids_cantidad_dict = Counter(pmids_lista)
    interacciones_cantidad_dict = Counter(interacciones_lista) # Clases y cantidad de ejemplos por clases ordenados de mayor a menor

    for interaccion, cantidad in interacciones_cantidad_dict.items():
        ifg_interaccion_lista = list()
        for i in range(0, len(pmids_lista), 1): # Agrega a "ifg_interaccion_lista" las ifg con la interacción "interaccion"
            if interacciones_lista[i] == interaccion:
                ifg_interaccion_lista.append([pmids_lista[i], genes_lista[i], drogas_lista[i], interacciones_lista[i]])
        if interaccion == "sin_interaccion":
            for ifg in random.sample(ifg_interaccion_lista, k=cantidad_ejemplos_sin_interaccion) :
                ifg_entrenamiento_lista.append(ifg)
        else:
            cantidad_ejemplos_prueba_clase_actual = math.ceil(cantidad*porcentaje_prueba) # ceil
            for ifg in random.sample(ifg_interaccion_lista, k=cantidad_ejemplos_prueba_clase_actual):
                ifg_prueba_lista.append(ifg)
                ifg_interaccion_lista.remove(ifg)
            for ifg in ifg_interaccion_lista:
                ifg_entrenamiento_lista.append(ifg)

    return ifg_entrenamiento_lista, ifg_prueba_lista

# ----------------------------------------------------------------------------------

def grafica_longitud_publicaciones(publicaciones_directorio):
    # Se cargan las publicaciones en un diccionario: publicaciones_dict[pmid] = contenido
    # print("Cargando diccionario de publicaciones.")
    publicaciones_dict = dict()
    publicaciones_en_directorio = os.listdir(publicaciones_directorio)
    for archivo in publicaciones_en_directorio:
        pmid = archivo.split(".")[0]
        archivo_ruta = os.path.join(publicaciones_directorio, archivo)
        with open(archivo_ruta, encoding="utf8") as publicacion:
            texto = publicacion.read()
            publicaciones_dict[pmid] = len(texto)

    longitudes = list(publicaciones_dict.values())
    longitudes.sort(reverse=True)
    media = statistics.mean(longitudes)
    print(longitudes[0])
    print(media)
    # print(sum(longitudes)/len(longitudes))
    # x = [i for i in range(len(longitudes))]
    # plt.plot([50000]*len(longitudes))
    plt.plot([media]*len(longitudes))
    plt.plot(longitudes)
    plt.savefig("distribucion_longitudes.png")
    plt.legend()
    plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------

def prueba_publicaciones(etiquetas_neural_networks_ruta,
                         publicaciones_directorio):

    # Se cargan las publicaciones en un diccionario: publicaciones_dict[pmid] = contenido
    print("Cargando diccionario de publicaciones.")
    publicaciones_dict = dict()
    publicaciones_en_directorio = os.listdir(publicaciones_directorio)
    for archivo in publicaciones_en_directorio:
        pmid = archivo.split(".")[0]
        archivo_ruta = os.path.join(publicaciones_directorio, archivo)
        with open(archivo_ruta, encoding="utf8") as publicacion:
            texto = publicacion.read()
            publicaciones_dict[pmid] = texto.split()
    print("Diccionario de publicaciones cargado.")

    # Se cargan las interacciones fármaco-gen en una lista
    print("Cargando las interacciones fármaco-gen en una lista.")
    pmids_lista = list()
    genes_lista = list()
    drogas_lista = list()
    interaccion_lista = list()
    with open(etiquetas_neural_networks_ruta, encoding="utf8") as etiquetas_neural_networks:
        lector_csv = csv.reader(etiquetas_neural_networks, delimiter=',', quoting=csv.QUOTE_ALL)
        for ifg in lector_csv:
            if ifg[3] != "sin_interaccion":
                pmids_lista.append(ifg[0])
                genes_lista.append(ifg[1])
                drogas_lista.append(ifg[2])
                interaccion_lista.append(ifg[3])
    print("Lista de interacciones fármaco-gen cargadas.")

    longitudes_lista = list()
    for pmid, contenido in publicaciones_dict.items():
        longitudes_lista.append(len(contenido))
    print(max(longitudes_lista))

if __name__ == "__main__":
    test_split("interacciones_lista.txt", "etiquetas_neural_networks_4_v3.csv", 0.2)
