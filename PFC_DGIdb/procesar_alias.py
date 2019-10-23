import sys

def aplanar(entrada_tsv,salida_tsv):
    '''
    Coloca el nombre del gen/droga y sus alias en la misma linea.
    El gen/droga queda en la primer posición.
    '''
    referencia = ''
    for linea in entrada_tsv:
        cadena_separada = linea.split("\t") # Separa la cadena en una lista de cadenas
        nombre = cadena_separada[0]
        alias = cadena_separada[1].strip() # Elimina el carácter "\n" del último elemento
        if referencia != nombre:
            salida_tsv.write("\n")
            salida_tsv.write("{}\t{}".format(nombre,alias))
            referencia = nombre
        else:
            salida_tsv.write("\t{}".format(alias))

def contar_alias(entrada_tsv):
    '''
    Cuenta la cantidad de alias por gen/droga.
    '''
    for linea in entrada_tsv:
        cadena_separada = linea.split("\t") # Separa la cadena en una lista de cadenas
        print(len(cadena_separada)-1)

def eliminar_duplicados(entrada_tsv,salida_tsv):
    '''
    Elimina los nombres/alias de genes/drogas duplicados en cada linea.
    Se mantiene el orden original.
    El nombre de gen/droga se encuentra en la primer posición.
    '''
    for linea in entrada_tsv:
        lista = linea.split("\t") # Separa la cadena en una lista de cadenas
        lista[-1] = lista[-1].strip() # Elimina el carácter "\n" del último elemento
        conjunto = set()
        contador = 0
        for elemento in lista: # Elimina los duplicados manteniendo el ordenamiento
            contador +=1
            if elemento not in conjunto:
                conjunto.add(elemento)
                if len(lista) == contador:
                    salida_tsv.write(elemento) # Si es el último elemento no coloca el "\t"
                else:
                    salida_tsv.write("{}\t".format(elemento)) # Si no es el último elemento coloca el "\t"
        salida_tsv.write("\n")

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Forma de uso: {} entrada salida".format(sys.argv[0]))
        exit()

    # archivo_entrada = open("D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\gen_alias.tsv",encoding="utf8")
    # archivo_salida = open("D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\gen_alias_aplanado.tsv",'w',encoding="utf8")
    # archivo_entrada = open("D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\droga_alias.tsv",encoding="utf8")
    # archivo_salida = open("D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\droga_alias_aplanado.tsv",'w',encoding="utf8")

    # aplanar(archivo_entrada,archivo_salida)

    # archivo_entrada = open("D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\gen_alias_aplanado.tsv",encoding="utf8")

    # contar_alias(archivo_entrada)

    # archivo_entrada = open("D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\gen_alias_aplanado.tsv",encoding="utf8")
    # archivo_salida = open("D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\gen_alias_entrenamiento.tsv",'w',encoding="utf8")
    # archivo_entrada = open("D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\droga_alias_aplanado.tsv",encoding="utf8")
    # archivo_salida = open("D:\Descargas\Python\PFC_DGIdb_src\PFC_DGIdb\droga_alias_entrenamiento.tsv",'w',encoding="utf8")

    # eliminar_duplicados(archivo_entrada,archivo_salida)

    # archivo_salida.close()
    # archivo_entrada.close()