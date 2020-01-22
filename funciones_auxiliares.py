from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from collections import Counter
import csv
import statistics
import random
import math

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
                     ejemplos_cantidad, # Cantidad de ejemplos a cargar
                     porcentaje_prueba): # Porcentaje de los ejemplos que se utilizarán para la prueba
    '''
    Devuelve (training, test): los conjuntos de interacciones fármaco-gen para
    entrenamiento y prueba balanceados en cantidad de ejemplos por clase.
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
            pmids_lista.append(linea[0])
            genes_lista.append(linea[1])
            drogas_lista.append(linea[2])
            if linea[3] not in interacciones_considerar:
                interacciones_lista.append("other")
            else:
                interacciones_lista.append(linea[3])

    # Armado de los conjuntos balanceados de entrenamiento y prueba
    interacciones_cantidad_dict = Counter(interacciones_lista)
    cantidad_clases = len(interacciones_cantidad_dict)
    if porcentaje_prueba == 0.0:
        cantidad_ejemplos_prueba_por_clase = 0
        cantidad_minima_ejemplos = 0
    else:
        cantidad_minima_ejemplos = math.ceil(cantidad_clases/porcentaje_prueba)
    if ejemplos_cantidad < cantidad_minima_ejemplos:
        ejemplos_cantidad = cantidad_minima_ejemplos
    cantidad_ejemplos_prueba_por_clase = int((ejemplos_cantidad*porcentaje_prueba)/cantidad_clases)
    cantidad_ejemplos_entrenamiento_por_clase = int((ejemplos_cantidad-(cantidad_ejemplos_prueba_por_clase*cantidad_clases))/cantidad_clases)
    ejemplos_por_clase = int(ejemplos_cantidad/cantidad_clases)
    seed = random.random()
    random.seed(seed)
    for interaccion, cantidad in interacciones_cantidad_dict.items():
        ifg_interaccion_lista = list()
        for i in range(0, len(pmids_lista), 1):
            if interacciones_lista[i] == interaccion:
                ifg_interaccion_lista.append([pmids_lista[i], genes_lista[i], drogas_lista[i], interacciones_lista[i]])
        if cantidad >= ejemplos_por_clase:
            for ifg in random.sample(ifg_interaccion_lista, k=cantidad_ejemplos_prueba_por_clase):
                ifg_balanceadas_prueba_lista.append(ifg)
                ifg_interaccion_lista.remove(ifg)
            for ifg in random.sample(ifg_interaccion_lista, k=cantidad_ejemplos_entrenamiento_por_clase):
                ifg_balanceadas_entrenamiento_lista.append(ifg)
        else:
            for ifg in random.choices(ifg_interaccion_lista, k=cantidad_ejemplos_prueba_por_clase):
                ifg_balanceadas_prueba_lista.append(ifg)
            for ifg in random.choices(ifg_interaccion_lista, k=cantidad_ejemplos_entrenamiento_por_clase):
                ifg_balanceadas_entrenamiento_lista.append(ifg)

    return ifg_balanceadas_entrenamiento_lista, ifg_balanceadas_prueba_lista

if __name__ == "__main__":
    etiquetas_archivo_ruta = "E:/Descargas/Python/PFC_DGIdb_src/etiquetas_neural_networks.csv"
    interacciones_lista_ruta = "E:/Descargas/Python/PFC_DGIdb_src/interacciones_lista2.txt"
    ifg_balanceadas_entrenamiento_lista, ifg_balanceadas_prueba_lista = balancear_clases(etiquetas_archivo_ruta, interacciones_lista_ruta, 10, 0.2)
    for ifg in ifg_balanceadas_entrenamiento_lista:
        print(ifg)
    print("xxxxxxxxxxxxxxxxxxxxxxx")
    for ifg in ifg_balanceadas_prueba_lista:
        print(ifg)
    