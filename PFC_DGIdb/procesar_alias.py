import sys
import csv

def eliminar_repetidos_gen(gen_ruta,alias_gen_ruta):
    '''
    Entradas:
    - Archivo csv con la lista de genes.
    - Archivo csv con la lista de alias para cada gen
    Salida: 
    - Archivo csv con tantas filas como genes hay, con el formato "gen seguido de aliases" sin denominaciones repetidas
    '''

    nombres_list = list()
    with open(gen_ruta,encoding="utf8") as genes:
        lector_csv = csv.reader(genes,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            nombres_list.append(fila[0])

    salida = open("E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/alias_gen.csv","w",encoding="utf8")
    escritor_csv = csv.writer(salida,delimiter=',',lineterminator="\n")
    for elemento in nombres_list:
        with open(alias_gen_ruta,encoding="utf8") as alias_gen:
            lector_csv = csv.reader(alias_gen,delimiter=',',quoting=csv.QUOTE_ALL)
            alias_lista = list()
            for fila in lector_csv:
                if elemento == fila[0]:
                    alias_lista.append(fila[1])
                    alias_lista.append(fila[2])
            lista = [elemento] + alias_lista
            lista = list(set(lista))
            lista.remove(elemento)
            lista.sort(reverse=True)
            lista.append(elemento)
            lista.reverse()
            escritor_csv.writerow(lista)
    salida.close()
        
def eliminar_repetidos_droga(droga_ruta,alias_droga_ruta1,alias_droga_ruta2):
    '''
    Entradas:
    - Archivo csv con la lista de drogas.
    - Archivo csv con la lista de alias1 para cada droga
    - Archivo csv con la lista de alias2 para cada droga
    Salida: 
    - Archivo csv con tantas filas como drogas hay, con el formato "droga seguido de aliases" sin denominaciones repetidas
    '''

    nombres_list = list()
    with open(droga_ruta,encoding="utf8") as drogas:
        lector_csv = csv.reader(drogas,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            nombres_list.append(fila[0])

    salida = open("E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/alias_droga.csv","w",encoding="utf8")
    escritor_csv = csv.writer(salida,delimiter=',',lineterminator="\n")
    for elemento in nombres_list:
        alias_lista = list()
        with open(alias_droga_ruta1,encoding="utf8") as alias_droga1:
            lector_csv = csv.reader(alias_droga1,delimiter=',',quoting=csv.QUOTE_ALL)
            for fila in lector_csv:
                if elemento == fila[0]:
                    alias_lista = alias_lista + fila[1:]
        with open(alias_droga_ruta2,encoding="utf8") as alias_droga2:
            lector_csv = csv.reader(alias_droga2,delimiter=',',quoting=csv.QUOTE_ALL)
            for fila in lector_csv:
                if elemento == fila[0]:
                    alias_lista = alias_lista + fila[1:]
        lista = [elemento] + alias_lista
        lista = list(set(lista))
        lista.remove(elemento)
        lista.sort(reverse=True)
        lista.append(elemento)
        lista.reverse()
        escritor_csv.writerow(lista)
    salida.close()

def formato_insercion_alias(alias_ruta):
    salida = open("E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/formato_insercion_alias_droga.csv","w",encoding="utf8")
    escritor_csv = csv.writer(salida,delimiter=',',lineterminator="\n")
    with open(alias_ruta,encoding="utf8") as alias:
        lector_csv = csv.reader(alias,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            nombre = fila[0]
            aliases = fila[1:]
            for elemento in aliases:
                lista = [nombre, elemento]
                escritor_csv.writerow(lista)
    salida.close()

def limpiar_alias(in_file, out_file):
    with open(out_file, "w", encoding="utf8") as salida:
        writer = csv.writer(salida)
        with open(in_file, encoding="utf8") as f:
            reader = csv.reader(f)
            for row in reader:
                # primer elemento es el nombre original
                all_alias = set(e.lower() for e in row[1:] if not e.isnumeric())
                writer.writerow([row[0].lower()] + sorted(all_alias))


def llenar_ids(in_file, out_file, ids_file):
    ids = {}
    with open(ids_file) as f:
        reader = csv.reader(f)
        for row in reader:
            ids[row[1]] = row[0]

    with open(out_file, "w", encoding="utf8") as salida:
        writer = csv.writer(salida)
        with open(in_file, encoding="utf8") as f:
            reader = csv.reader(f)
            for row in reader:
                writer.writerow([ids[row[0]]] + row)



# def _limpiar(alias):
#     alias = ungreek._ungreek(alias)
#     alias = ungreek._clean_html(alias)
#     return alias

# def limpiar_alias(in_file, out_file):
#     """Dado un archivo de entrada donde cada fila es un gen/droga con uno de sus alias,
#     arma un archivo de salida donde agrupa en cada fila todos los alias. Además los limpia."""
#     alias = {}
#     with open(in_file, encoding="utf8") as f:
#         reader = csv.reader(f)
#         for row in reader:
#             # primer elemento es el ID, segundo el nombre original, tercero el alias
#             # entidad = alias.setdefault(row[0], [row[1]]) # inicializo con una lista con el nombre
#             entidad = alias.setdefault(row[1].lower(), [])
#             entidad.append(row[2])

#     # transformo el diccionario en una lista de listas:
#     list_alias = []
#     for k, v in alias.items():
#         all_alias = set(_limpiar(e) for e in v if not e.isnumeric())
#         list_alias.append([k] + sorted(all_alias))

#     with open(out_file, "w", encoding="utf8") as salida:
#         writer = csv.writer(salida)
#         writer.writerows(list_alias)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Forma de uso: {} entrada salida".format(sys.argv[0]))
        exit()

    limpiar_alias(sys.argv[1], sys.argv[2])

    # gen_ruta = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/dgidb_export_genes.csv"
    # alias_gen_ruta = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/dgidb_export_alias_gen.csv"
    # eliminar_repetidos_gen(gen_ruta,alias_gen_ruta)

    # droga_ruta = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/dgidb_export_drogas.csv"
    # alias_droga_ruta1 = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/dgidb_export_alias_droga1.csv"
    # alias_droga_ruta2 = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/dgidb_export_alias_droga2.csv"
    # eliminar_repetidos_droga(droga_ruta,alias_droga_ruta1,alias_droga_ruta2)

    # alias_gen_ruta = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/alias_gen.csv"
    # formato_insercion_alias(alias_gen_ruta)

    # alias_droga_ruta = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/alias_droga.csv"
    # formato_insercion_alias(alias_droga_ruta)

    # llenar_ids("alias_gen.csv", "alias_gen_2.csv", "ids_gen.csv")
    # llenar_ids("alias_droga.csv", "alias_droga_2.csv", "ids_droga.csv")
