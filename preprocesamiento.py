from datos_preprocesamiento import cargar_publicaciones, cargar_abstracts, cargar_ocurrencias, cargar_aliases_dict, cargar_etiquetas_dict, cargar_pmids
import pathlib
import sys
import argparse
import logging
import random
import re

def unir_elementos_lista(lista, posicion_inicio, cantidad_elementos):
    '''
    Agrupa los elementos de una lista con espacio entre ellos.
    La cantidad de elementos a agrupar depende del parámetro cantidad_elementos
    Mínimo: 1, deja la lista tal cual entra.
    Los elementos anteriores a la posicion_inicio se dejan como están
    '''
    principio = lista[:posicion_inicio]
    fin = list()
    for i in range(posicion_inicio, len(lista), cantidad_elementos):
        fin.append(" ".join(lista[i:i+cantidad_elementos]))
    lista_salida = principio + fin
    return lista_salida

def aplanar_lista(lista):
    '''
    Separa todos los elementos de una lista por espacios
    '''
    lista_salida = list()
    for elemento in lista:
        lista_salida = lista_salida + elemento.split()
    return lista_salida

def lista_a_cadena(lista):
    '''
    Convierte una lista en una cadena con sus elementos separados por espacios
    '''
    cadena = " "
    return cadena.join(lista)

def remplazo_inteligente(texto,cadena_a_remplazar,remplazar_con):
    '''
    Remplaza palabras completas o palabras simultáneas pero no subcadenas.
    '''
    texto_lista = texto.split()
    maxima_longitud = 33 # Longitud del alias más largo. Encontrada con el algortimo longitud_maxima_alias del archivo datos_preprocesamiento.py
    for cantidad_elementos in range(1, maxima_longitud, 1):
        for posicion_inicio in range(0, maxima_longitud, 1):
            lista = unir_elementos_lista(texto_lista,posicion_inicio,cantidad_elementos)
            for posicion, elemento in enumerate(lista):
                # print("Posicion: {}; Elemento: {}".format(posicion,elemento))
                if elemento == cadena_a_remplazar:
                    # print("Posicion: {}; Elemento: {}".format(posicion,elemento))
                    lista[posicion] = remplazar_con
                    texto_lista = aplanar_lista(lista)
    texto_remplazado = lista_a_cadena(texto_lista)
    return texto_remplazado

def reemplazar_bme(pmid, contents, output_dir):
    """
    contents: el contenido del paper (puede ser todo el artículo, o el abstract o título o palabras clave)
    output_dir: directorio de salida
    Variables disponibles:
    alias_gen: diccionario alias -> [nombres reales de genes con ese alias]
    alias_droga: diccionario alias -> [nombres reales de drogas con ese alias]
    etiquetas_genes: dicciona pmid -> [genes etiquetados en ese artículo]
    etiquetas_drogas: dicciona pmid -> [drogas etiquetadas en ese artículo]
    """

    # REEMPLAZO PARA GENES -----------------------------------------------------------
    
    genes_etiquetados = list(etiquetas_genes[pmid])
    genes_etiquetados_encontrados = set()
    
    for og in ocurrencias_genes_se_sr[pmid]: # analizo sin embedding - sin repetición
        genes = alias_gen[og]
        if len(genes) != 1:
            logging.error("Usando SIN REPETICIONES se encontró un alias de gen repetido!!: %s", og)
            logging.error("Diccionario: %s", str(genes))
        gen = list(genes)[0]
        contents = contents.replace(og, "{[" + gen + "]}")
        if gen in genes_etiquetados:
            genes_etiquetados_encontrados.add(gen)
    
    if set(genes_etiquetados) != genes_etiquetados_encontrados:
        # analizo sin embedding - con repetición
        ocurrencias_repetidas_no_analizadas = set(ocurrencias_genes_se_cr[pmid]) - set(ocurrencias_genes_se_sr[pmid])
        for og in ocurrencias_repetidas_no_analizadas:
            genes = alias_gen[og]
            gen = None
            for g in genes: # si el alias se mapea a varios genes, tratar de elegir uno etiquetado
                if g in genes_etiquetados:
                    gen = g
                    logging.info("La ocurrencia %s tiene varios nombres reales de genes (%s), se eligió: %s por estar etiquetado",
                                og, str(genes), gen)
                    genes_etiquetados_encontrados.add(gen)
                    break
            if not gen:
                # ninguno etiquetado, elijo cualquiera?
                gen = list(genes)[random.randint(0, len(genes)-1)]
                logging.warning("La ocurrencia %s tiene varios nombres reales de genes (%s), se eligió aleatoriamente: %s",
                                og, str(genes), gen)
            contents = contents.replace(og, "{[" + gen + "]}")
            
            if set(genes_etiquetados) != genes_etiquetados_encontrados:
                # analizo con embedding - con repetición
                genes_etiquetados_no_encontrados = set(genes_etiquetados) - genes_etiquetados_encontrados
                ocurrencias_repetidas_no_analizadas = set(ocurrencias_genes_ce_cr[pmid]) - set(ocurrencias_genes_se_cr[pmid])
                for og in ocurrencias_repetidas_no_analizadas:
                    logging.info("Analizando ahora gen con embedding - con repetición...")
                    genes = alias_gen[og]
                    gen = None
                    for g in genes: # si el alias se mapea a varios genes, tratar de elegir uno etiquetado
                        if g in genes_etiquetados_no_encontrados:
                            gen = g
                            logging.info("La ocurrencia %s tiene varios nombres reales de genes (%s), se eligió: %s por estar etiquetado",
                                        og, str(genes), gen)
                            genes_etiquetados_encontrados.add(gen)
                            contents = contents.replace(og, "{[" + gen + "]}")
                            break

    genes_etiquetados_no_encontrados = set(genes_etiquetados) - genes_etiquetados_encontrados
    logging.info("De los genes etiquetados, no se encontró: %s", str(genes_etiquetados_no_encontrados))
    
    
    
    # REEMPLAZO PARA DROGAS -----------------------------------------------------------

    drogas_etiquetadas = list(etiquetas_drogas[pmid])
    drogas_etiquetadas_encontradas = set()
    
    for og in ocurrencias_drogas_se_sr[pmid]: # analizo sin embedding - sin repetición
        drogas = alias_droga[og]
        if len(drogas) != 1:
            logging.error("Usando SIN REPETICIONES se encontró un alias de droga repetido!!: %s", og)
            logging.error("Diccionario: %s", str(drogas))
        droga = list(drogas)[0]
        contents = contents.replace(og, "{[" + droga + "]}")
        if droga in drogas_etiquetadas:
            drogas_etiquetadas_encontradas.add(droga)
    
    if set(drogas_etiquetadas) != drogas_etiquetadas_encontradas:
        # analizo sin embedding - con repetición
        ocurrencias_repetidas_no_analizadas = set(ocurrencias_drogas_se_cr[pmid]) - set(ocurrencias_drogas_se_sr[pmid])
        for og in ocurrencias_repetidas_no_analizadas:
            drogas = alias_droga[og]
            droga = None
            for d in drogas: # si el alias se mapea a varios drogas, tratar de elegir uno etiquetado
                if d in drogas_etiquetadas:
                    droga = d
                    logging.info("La ocurrencia %s tiene varios nombres reales de drogas (%s), se eligió: %s por estar etiquetado",
                                og, str(drogas), droga)
                    drogas_etiquetadas_encontradas.add(droga)
                    break
            if not droga:
                # ninguno etiquetado, elijo cualquiera?
                droga = list(drogas)[random.randint(0, len(drogas)-1)]
                logging.warning("La ocurrencia %s tiene varios nombres reales de drogas (%s), se eligió aleatoriamente: %s",
                                og, str(drogas), droga)
            contents = contents.replace(og, "{[" + droga + "]}")
            
            if set(drogas_etiquetadas) != drogas_etiquetadas_encontradas:
                # analizo con embedding - con repetición
                drogas_etiquetadas_no_encontradas = set(drogas_etiquetadas) - drogas_etiquetadas_encontradas
                ocurrencias_repetidas_no_analizadas = set(ocurrencias_drogas_ce_cr[pmid]) - set(ocurrencias_drogas_se_cr[pmid])
                for og in ocurrencias_repetidas_no_analizadas:
                    logging.info("Analizando droga ahora con embedding - con repetición...")
                    drogas = alias_droga[og]
                    droga = None
                    for g in drogas: # si el alias se mapea a varios drogas, tratar de elegir uno etiquetado
                        if g in drogas_etiquetadas_no_encontradas:
                            droga = g
                            logging.info("La ocurrencia %s tiene varios nombres reales de drogas (%s), se eligió: %s por estar etiquetado",
                                        og, str(drogas), droga)
                            drogas_etiquetadas_encontradas.add(droga)
                            contents = contents.replace(og, "{[" + droga + "]}")
                            break

    drogas_etiquetadas_no_encontradas = set(drogas_etiquetadas) - drogas_etiquetadas_encontradas
    logging.info("De las drogas etiquetadas, no se encontró: %s", str(drogas_etiquetadas_no_encontradas))
    
    with open(output_dir / (pmid + ".txt"), "w") as f:
        f.write(contents)



if __name__ == "__main__":
    logging.getLogger("datos_preprocesamiento").setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("lista_pmid", help="Archivo con el listado de PMID para trabajar")
    parser.add_argument("dir_txt_publicaciones", help="Directorio con los TXT de los papers")
    parser.add_argument("ruta_abstracts", help="Archivo con los abstracts/títulos/palabras clave")
    parser.add_argument("dir_pfc_dgidb", help="Directorio PFC_DGIdb para obtener archivos con datos")
    parser.add_argument("output_dir", help="Directorio de salida con los textos reemplazados")
    args = parser.parse_args()

    print("Cargando todo...")

    abstracts_dict = cargar_abstracts(args.ruta_abstracts)
    publicaciones_dict = cargar_publicaciones(args.dir_txt_publicaciones, abstracts_dict,
                                              cargar_pmids(args.lista_pmid))
    
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()


    dir_pfc_dgidb = pathlib.Path(args.dir_pfc_dgidb)
    alias_gen = cargar_aliases_dict(dir_pfc_dgidb / "alias_gen.csv")
    alias_droga = cargar_aliases_dict(dir_pfc_dgidb / "alias_droga.csv")

    ocurrencias_genes_se_sr = cargar_ocurrencias("ocurrencias_genes_se_sr.csv")
    ocurrencias_genes_se_cr = cargar_ocurrencias("ocurrencias_genes_se_cr.csv")
    ocurrencias_genes_todas = cargar_ocurrencias("ocurrencias_genes_todas.csv")
    ocurrencias_drogas_se_sr = cargar_ocurrencias("ocurrencias_drogas_se_sr.csv")
    ocurrencias_drogas_se_cr = cargar_ocurrencias("ocurrencias_drogas_se_cr.csv")
    ocurrencias_drogas_todas = cargar_ocurrencias("ocurrencias_drogas_todas.csv")

    etiquetas_genes, etiquetas_drogas = cargar_etiquetas_dict(dir_pfc_dgidb / "pfc_dgidb_export_ifg.csv")

    for pmid, contents in publicaciones_dict.items():
        reemplazar_bme(pmid, contents, output_dir)

    print("Listo.")