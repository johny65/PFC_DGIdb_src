from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np

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