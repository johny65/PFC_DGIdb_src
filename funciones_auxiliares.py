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

def kfolding(particiones_numero, x_entrenamiento):
    ejemplos_cantidad = len(x_entrenamiento)
    longitud = 0
    folds_dict = dict()
    if (ejemplos_cantidad % particiones_numero) == 0:
        longitud = len(x_entrenamiento)
    else:
        longitud = int(ejemplos_cantidad/particiones_numero)*particiones_numero
    ejemplos_validacion_cantidad = longitud/particiones_numero
    for i in range(0,particiones_numero,1):
        indices_validacion = np.arange(int(i*ejemplos_validacion_cantidad),int(i*ejemplos_validacion_cantidad+ejemplos_validacion_cantidad))
        indices_entrenamiento = list()
        for j in range(0,longitud,1):
            if j not in indices_validacion:
                indices_entrenamiento.append(j)
        indices_entrenamiento = np.asarray(indices_entrenamiento)
        folds_dict[i] = [indices_entrenamiento, indices_validacion]
    return folds_dict
