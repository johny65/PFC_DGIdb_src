import sys
import csv
import numpy as np
import re
import os

# Preprocesamiento de los datos (texto)
# from keras.preprocessing import text,sequence 

def cargar_pmids(pmids_etiquetas_completas_csv): # Carga los pmids con las etiquetas completas
    pmids_lista = list()
    with open(pmids_etiquetas_completas_csv,encoding="utf8") as pmids_csv:
        lector_csv = csv.reader(pmids_csv,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            pmids_lista.append(fila[0])
    return pmids_lista

def cargar_aliases(aliases_ruta): # Carga los aliases de gen o de droga dependiendo del archivo de entrada
    aliases_conjunto = set()
    with open(aliases_ruta,encoding="utf8") as aliases_csv:
        lector_csv = csv.reader(aliases_csv,delimiter=',',quoting=csv.QUOTE_ALL)
        for fila in lector_csv:
            for e in fila:
                aliases_conjunto.add(e.lower())
    return aliases_conjunto

def aliases_repeticiones(aliases_conjunto,aliases_ruta,salida):
    with open(salida,"w",encoding="utf8") as salida_csv:
        escritor_csv = csv.writer(salida_csv,delimiter=',',lineterminator="\n")
        for elemento in aliases_conjunto:
            lista = list()
            contador = 0
            with open(aliases_ruta,encoding="utf8") as aliases_csv:
                lector_csv = csv.reader(aliases_csv,delimiter=',',quoting=csv.QUOTE_ALL)
                for fila in lector_csv:
                    if elemento in fila:
                        contador += 1
                    if contador>1:
                        lista.append(elemento)
                        escritor_csv.writerow(lista)
                        break

def cargar_embeddings(embeddings_ruta): # Carga los vectores de embeddings de GloVe
    return

def cargar_publicaciones(publicaciones_directorio,pmids_titulos_abstracts_keywords_ruta,pmids_lista): # Carga las publicaciones que ya se encuentran en formato txt
    publicaciones_dict = dict()
    for pmid in pmids_lista:
        archivo_nombre = pmid + ".txt"
        if archivo_nombre in sorted(os.listdir(publicaciones_directorio)):
            archivo_ruta = os.path.join(publicaciones_directorio,archivo_nombre)
            with open(archivo_ruta,encoding="utf8") as publicacion:
                publicaciones_dict[pmid] = publicacion.read().lower()
        else:
            with open(pmids_titulos_abstracts_keywords_ruta,encoding="utf8") as abstracts:
                lector_csv = csv.reader(abstracts,delimiter=',',quoting=csv.QUOTE_ALL)
                for linea in lector_csv:
                    if linea[0] == pmid:
                        publicaciones_dict[pmid] = linea[1].lower()
    return publicaciones_dict

def ocurrencias(entidades,publicaciones,salida): # Muestra las apariciones de genes/aliases o drogas/aliases en las publicaciones
    with open(salida,"w",encoding="utf8") as salida_csv:
        escritor_csv = csv.writer(salida_csv,delimiter=',',lineterminator="\n")
        for elemento in publicaciones:
            lista = list()
            for entidad in entidades:
                if entidad in publicaciones[elemento]:
                    lista.append(entidad)
            escritor_csv.writerow(lista)

### ---

def cargar_datos_entrenamiento(interaccion_farmaco_gen,alias_gen_entrenamiento,alias_droga_entrenamiento,embeddings_ruta):
    '''

    '''

    # embeddings_dict = dict()
    # with open(embeddings_ruta,encoding="utf8") as embeddings:
    #     for fila in embeddings:
    #         datos = fila.split()
    #         palabra = datos[0]
    #         embedding = np.asarray(datos[1:],dtype='float32')
    #         embeddings_dict[palabra] = embedding

    # print("Embeddings cargados")

    pubs_dict = dict() # Mapa / Diccionario
    # abstracts_list = list() # list() == []
    with open(datos_entrenamiento_ruta,encoding="utf8") as archivo_tsv:
        lector_tsv = csv.reader(archivo_tsv, delimiter='\t', quoting=csv.QUOTE_ALL)
        for fila in lector_tsv:
            # 0: pmid, 1: gen, 2: droga, 3: interacción, 4: abstract
            s = fila[4] #.lower()
            # s = re.sub('[\.\,]','',s)
            abstracts_dict[fila[0]] = s # abstracts_dict['pmid'] = abstract.str
            # abstracts_list.append(fila[4])

    ruta_txt = "E:/Descargas/Python/PFC_DGIdb_src/scraping/files/txt"
    for archivo in sorted(os.listdir(ruta_pdf)):
        if archivo.endswith(".txt") and os.path.splittext(archivo)[0] in alias_gen_conjunto:
            entrada_ruta = os.path.join(ruta_pdf,archivo)
            salida_nombre = archivo.replace(".pdf",".txt")
            salida_ruta = os.path.join(ruta_txt,salida_nombre)
            lista = [xpdf_ruta,"-enc","UTF-8","-nopgbrk","-nodiag",entrada_ruta,salida_ruta]
            subprocess.Popen(lista)
            # res = subprocess.run(lista)
            # if res.returncode != 0:
            #     print("Error: " + archivo)

    # print(abstracts_dict['17288876'])
    # print(abstracts_dict)
    # print(abstracts_list[4])

    print("Abstracts cargados")

    aparicion_gen = open("aparicion_gen.csv","w",encoding="utf8")
    aparicion_droga = open("aparicion_droga.csv","w",encoding="utf8")
    escritor_csv_gen = csv.writer(aparicion_gen,delimiter=',',lineterminator="\n")
    escritor_csv_droga = csv.writer(aparicion_droga,delimiter=',',lineterminator="\n")

    for fila in abstracts_dict:
        palabras = abstracts_dict[fila].split()
        lista_gen = list()
        lista_droga = list()
        for palabra in palabras:
            if palabra in alias_gen_conjunto:
                lista_gen.append(palabra)
            if palabra in alias_droga_conjunto:
                lista_droga.append(palabra)
        escritor_csv_gen.writerow(lista_gen)
        escritor_csv_droga.writerow(lista_droga)

    aparicion_gen.close()
    aparicion_droga.close()

    print("Archivos de búsqueda de aliases creados")
    
    # sin_embedding = open("sin_embedding.csv","w",encoding="utf8")
    # escritor_tsv = csv.writer(sin_embedding,delimiter=',',lineterminator="\n",quoting=csv.QUOTE_ALL)

    # for fila in abstracts_dict:
    #     palabras = abstracts_dict[fila].split()
    #     lista = list()
    #     for palabra in palabras:
    #         if palabra not in embeddings_dict:
    #             lista.append(palabra)
    #     escritor_tsv.writerow(lista)

    # sin_embedding.close()

    # print("Archivo sin_embedding creado")

    # lista_gen = list()
    # lista_droga = list()
    # for fila in abstracts_dict:
    #     palabras = abstracts_dict[fila].split()
        
        # g = []
        # for gen in alias_gen_conjunto:
        #     if gen in palabras:
        #         g.append(gen)
        # lista_gen.append(g)

        # d = []
        # for droga in alias_droga_conjunto:
        #     if droga in palabras:
        #         d.append(droga)
        # lista_droga.append(d)

    # print(lista_gen)
    # print(lista_droga)

    # Crea el vocabulario a partir de los datos de entrenamiento
    # tokenizer = text.Tokenizer()
    # tokenizer.fit_on_texts(abstracts_list)

    # print(len(tokenizer.word_index))
    # print(tokenizer.document_count)

    # # Vectorización de los datos de entrada (entrenamiento y prueba): "Hola querido mundo" -> [3857 274 982]
    # x_entrenamiento_vectorizado = tokenizer.texts_to_sequences(x_entrenamiento)
    # x_prueba_vectorizado = tokenizer.texts_to_sequences(x_prueba)

    # # Obtener longitud del ejemplo más largo
    # maxima_longitud = len(max(x_entrenamiento_vectorizado,key=len))
    # if maxima_longitud > MAXIMA_LONGITUD_EJEMPLOS:
    #     maxima_longitud = MAXIMA_LONGITUD_EJEMPLOS

    # # Se arreglan los ejemplos para que todos tengan la misma longitud
    # x_entrenamiento_arreglado = sequence.pad_sequences(x_entrenamiento_vectorizado,maxlen=maxima_longitud)
    # x_prueba_arreglado = sequence.pad_sequences(x_prueba_vectorizado,maxlen=maxima_longitud)

    # return

# def interacciones_tipos(datos_entrenamiento):
#     interacciones = set()
#     for linea in datos_entrenamiento:
#         datos = linea.split("\t")
#         # pmid = datos[0]
#         # gen = datos[1]
#         # droga = datos[2]
#         interaccion = datos[3]
#         # abstract = datos[4:]
#         interacciones.add(interaccion)
#     # interacciones_mapa = list(interacciones).sort()
#     interacciones_mapa = list(interacciones)
#     interacciones_mapa.sort()
#     return interacciones_mapa

# def incorporacion_glove():
#     '''
#     Carga los vectores de embedding preentrenados de GloVe
#     '''
    
#     embeddings_index = dict()
#     f = open('glove.6B.300d.txt',encoding="utf8")
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#     f.close()

#     embedding_matrix = np.zeros((TOP_PALABRAS_FRECUENTES,DIMENSION_VECTORES_EMBEDDING))
#     for word, index in tokenizer.word_index.items():
#         if index > TOP_PALABRAS_FRECUENTES - 1:
#             break
#         else:   
#             embedding_vector = embeddings_index.get(word)
#             if embedding_vector is not None:
#                 embedding_matrix[index] = embedding_vector  


# def sample_to_number():

#     return 

if __name__ == "__main__":
    # if len(sys.argv) != 1:
    #     print("Forma de uso: {} entrada salida".format(sys.argv[0]))
    #     exit()
    
    # pmids_etiquetas_completas_csv = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/pmids_etiquetas_completas.csv"
    # pmids_lista = cargar_pmids(pmids_etiquetas_completas_csv)
    # print(pmids_lista)
    # print("pmids cargados")

    aliases_gen_ruta = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/alias_gen.csv"
    aliases_gen = cargar_aliases(aliases_gen_ruta)
    print(aliases_gen)
    print("alias gen cargados")

    aliases_droga_ruta = "E:/Descargas/Python/PFC_DGIdb_src/PFC_DGIdb/alias_droga.csv"
    aliases_droga = cargar_aliases(aliases_droga_ruta)
    # print(aliases_droga)
    # print("alias droga cargados")

    # publicaciones_directorio = "E:/Descargas/Python/PFC_DGIdb_src/scraping/files/txt"
    # pmids_titulos_abstracts_keywords_ruta = "E:/Descargas/Python/PFC_DGIdb_src/scraping/pmids_titulos_abstracts_keywords.csv"
    # publicaciones_dict = cargar_publicaciones(publicaciones_directorio,pmids_titulos_abstracts_keywords_ruta,pmids_lista)
    # print(publicaciones_dict)
    # print("publicaciones cargadas")

    # ocurrencias(aliases_gen,publicaciones_dict,"ocurrencias_genes.csv")
    # ocurrencias(aliases_droga,publicaciones_dict,"ocurrencias_drogas.csv")

    # aliases_repeticiones(aliases_gen,aliases_gen_ruta,"repeticiones_genes.csv")
    # aliases_repeticiones(aliases_droga,aliases_droga_ruta,"repeticiones_drogas.csv")