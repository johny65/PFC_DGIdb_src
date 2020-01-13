from datos_preprocesamiento import cargar_publicaciones, cargar_abstracts, cargar_ocurrencias,\
                                   cargar_aliases_dict, cargar_etiquetas_dict, cargar_pmids
import pathlib
import sys
import argparse
import logging
import random
import re

WRAPPER_INI = "xxx"
WRAPPER_FIN = "xxx"


def reemplazo_inteligente(texto,cadena_a_remplazar,remplazar_con):
    return re.sub(r"\b{}\b".format(re.escape(cadena_a_remplazar)), remplazar_con, texto)

def replace(original_text, search_for_replace, replace_with):
    """Wrapper a la función de reemplazo."""
    logging.info("Reemplazando %s por %s", search_for_replace, replace_with)
    # reemplazo básico:
    # return original_text.replace(search_for_replace, replace_with)
    # reemplazo inteligente con regex:
    return reemplazo_inteligente(original_text, search_for_replace, replace_with)

def reemplazar_bme(pmid, contents, output_dir):
    """
    contents: el contenido del paper (puede ser todo el artículo, o el abstract o título o palabras clave)
    output_dir: directorio de salida
    Variables disponibles:
    alias_gen: diccionario alias -> [nombres reales de genes con ese alias]
    alias_droga: diccionario alias -> [nombres reales de drogas con ese alias]
    etiquetas_genes: diccionario pmid -> [genes etiquetados en ese artículo]
    etiquetas_drogas: diccionario pmid -> [drogas etiquetadas en ese artículo]
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
        contents = replace(contents, og, WRAPPER_INI + gen + WRAPPER_FIN)
        if gen in genes_etiquetados and og == gen:
            # sólo lo marco como que lo encontré si una de las ocurrencias es el nombre real, sino no así
            # en otro paso reemplazo el nombre real sí o sí
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
                    if og == gen:
                        genes_etiquetados_encontrados.add(gen)
                    break
            if not gen:
                # ninguno etiquetado, elijo cualquiera?
                gen = list(genes)[random.randint(0, len(genes)-1)]
                logging.warning("La ocurrencia %s tiene varios nombres reales de genes (%s), se eligió aleatoriamente: %s",
                                og, str(genes), gen)
            contents = replace(contents, og, WRAPPER_INI + gen + WRAPPER_FIN)
            
            if set(genes_etiquetados) != genes_etiquetados_encontrados:
                # analizo con embedding - con repetición
                genes_etiquetados_no_encontrados = set(genes_etiquetados) - genes_etiquetados_encontrados
                ocurrencias_repetidas_no_analizadas = set(ocurrencias_genes_todas[pmid]) - set(ocurrencias_genes_se_cr[pmid])
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
                            contents = replace(contents, og, WRAPPER_INI + gen + WRAPPER_FIN)
                            break

    genes_etiquetados_no_encontrados = set(genes_etiquetados) - genes_etiquetados_encontrados
    if genes_etiquetados_no_encontrados:
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
        contents = replace(contents, og, WRAPPER_INI + droga + WRAPPER_FIN)
        if droga in drogas_etiquetadas and og == droga:
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
                    if og == droga:
                        drogas_etiquetadas_encontradas.add(droga)
                    break
            if not droga:
                # ninguno etiquetado, elijo cualquiera?
                droga = list(drogas)[random.randint(0, len(drogas)-1)]
                logging.warning("La ocurrencia %s tiene varios nombres reales de drogas (%s), se eligió aleatoriamente: %s",
                                og, str(drogas), droga)
            contents = replace(contents, og, WRAPPER_INI + droga + WRAPPER_FIN)
            
            if set(drogas_etiquetadas) != drogas_etiquetadas_encontradas:
                # analizo con embedding - con repetición
                drogas_etiquetadas_no_encontradas = set(drogas_etiquetadas) - drogas_etiquetadas_encontradas
                ocurrencias_repetidas_no_analizadas = set(ocurrencias_drogas_todas[pmid]) - set(ocurrencias_drogas_se_cr[pmid])
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
                            contents = replace(contents, og, WRAPPER_INI + droga + WRAPPER_FIN)
                            break

    drogas_etiquetadas_no_encontradas = set(drogas_etiquetadas) - drogas_etiquetadas_encontradas
    if drogas_etiquetadas_no_encontradas:
        logging.info("De las drogas etiquetadas, no se encontró: %s", str(drogas_etiquetadas_no_encontradas))
    
    with open(output_dir / (pmid + ".txt"), "w") as f:
        f.write(contents)



if __name__ == "__main__":
    logging.basicConfig(logging.INFO)
    # logging.getLogger("datos_preprocesamiento").setLevel(logging.INFO)

    lista_pmid = "PFC_DGIdb/pmids_etiquetas_completas.csv"
    dir_txt_publicaciones = "scraping/files/labeled/txt_ungreek"
    ruta_abstracts = "scraping/pmids_titulos_abstracts_keywords.csv"
    dir_pfc_dgidb = "PFC_DGIdb"

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directorio de salida con los textos reemplazados")
    args = parser.parse_args()

    print("Cargando todo...")

    abstracts_dict = cargar_abstracts(ruta_abstracts)
    publicaciones_dict = cargar_publicaciones(dir_txt_publicaciones, abstracts_dict,
                                              cargar_pmids(lista_pmid))
    
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    dir_pfc_dgidb = pathlib.Path(args.dir_pfc_dgidb)
    alias_gen = cargar_aliases_dict(dir_pfc_dgidb / "alias_gen.csv")
    alias_droga = cargar_aliases_dict(dir_pfc_dgidb / "alias_droga.csv")

    ocurrencias_genes_se_sr = cargar_ocurrencias("g_se_sr.csv")
    ocurrencias_genes_se_cr = cargar_ocurrencias("g_se_cr.csv")
    ocurrencias_genes_todas = cargar_ocurrencias("g_ce_cr.csv")
    ocurrencias_drogas_se_sr = cargar_ocurrencias("d_se_sr.csv")
    ocurrencias_drogas_se_cr = cargar_ocurrencias("d_se_cr.csv")
    ocurrencias_drogas_todas = cargar_ocurrencias("d_ce_cr.csv")

    etiquetas_genes, etiquetas_drogas = cargar_etiquetas_dict(dir_pfc_dgidb / "pfc_dgidb_export_ifg.csv")

    print("Listo carga.")

    for pmid, contents in publicaciones_dict.items():
        reemplazar_bme(pmid, contents, output_dir)
