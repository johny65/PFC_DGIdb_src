from datos_preprocesamiento import cargar_publicaciones, cargar_abstracts, cargar_ocurrencias,\
                                   cargar_aliases_dict, cargar_etiquetas_dict, cargar_pmids
import pathlib
import sys
import argparse
import logging
import random
import re

WRAPPER_INI = " xxx"
WRAPPER_FIN = "xxx "


def reemplazo_inteligente(texto, cadena_a_remplazar, remplazar_con):
    return re.sub(r"\b{}\b".format(re.escape(cadena_a_remplazar)), remplazar_con, texto)

def replace(original_text, search_for_replace, replace_with, pmid):
    """Wrapper a la función de reemplazo."""
    logging.info("Pmid %s: Reemplazando %s por %s", pmid, search_for_replace, replace_with)
    # reemplazo básico:
    # return original_text.replace(search_for_replace, replace_with)
    # reemplazo inteligente con regex:
    if isinstance(replace_with, list):
        replace_with = " ".join([WRAPPER_INI + e + WRAPPER_FIN for e in replace_with])
    else:
        replace_with = WRAPPER_INI + replace_with + WRAPPER_FIN
    return reemplazo_inteligente(original_text, search_for_replace, replace_with)


def reemplazar_bme(pmid, contents, output_dir):
    full_reemplazar_bme(pmid, contents, alias_gen, alias_droga, etiquetas_genes, etiquetas_drogas,
                        ocurrencias_genes_se_sr, ocurrencias_genes_se_cr, ocurrencias_genes_todas,
                        ocurrencias_drogas_se_sr, ocurrencias_drogas_se_cr, ocurrencias_drogas_todas,
                        output_dir, True)

def full_reemplazar_bme(pmid, contents, alias_gen, alias_droga, etiquetas_genes, etiquetas_drogas,
                        ocurrencias_genes_se_sr, ocurrencias_genes_se_cr, ocurrencias_genes_todas,
                        ocurrencias_drogas_se_sr, ocurrencias_drogas_se_cr, ocurrencias_drogas_todas,
                        output_dir, todisk):
    """
    contents: el contenido del paper (puede ser todo el artículo, o el abstract o título o palabras clave)
    output_dir: directorio de salida (si todisk = True)
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
        contents = replace(contents, og, gen, pmid)
        if gen in genes_etiquetados:
            genes_etiquetados_encontrados.add(gen)
    
    if set(genes_etiquetados) != genes_etiquetados_encontrados:
        # analizo sin embedding - con repetición
        ocurrencias_repetidas_no_analizadas = set(ocurrencias_genes_se_cr[pmid]) - set(ocurrencias_genes_se_sr[pmid])
        for og in ocurrencias_repetidas_no_analizadas:
            genes = alias_gen[og]
            reemplazar_por = []
            for g in genes: # si el alias se mapea a varios genes, tratar de elegir los etiquetados
                if g in genes_etiquetados:
                    reemplazar_por.append(g)
                    genes_etiquetados_encontrados.add(g)
            if not reemplazar_por:
                # ninguno etiquetado, elijo todos
                reemplazar_por = genes
                logging.warning("Pmid %s: La ocurrencia %s tiene varios nombres reales de genes (%s), se eligieron todos: %s",
                                pmid, og, str(genes), str(reemplazar_por))
            else:
                logging.info("Pmid %s: La ocurrencia %s tiene varios nombres reales de genes (%s), se eligió: %s por estar etiquetados",
                            pmid, og, str(genes), str(reemplazar_por))
            
            contents = replace(contents, og, reemplazar_por, pmid)
            
            if set(genes_etiquetados) != genes_etiquetados_encontrados:
                # analizo con embedding - con repetición
                genes_etiquetados_no_encontrados = set(genes_etiquetados) - genes_etiquetados_encontrados
                ocurrencias_repetidas_no_analizadas = set(ocurrencias_genes_todas[pmid]) - set(ocurrencias_genes_se_cr[pmid])
                logging.info("Pmid %s: Analizando ahora gen con embedding - con repetición...", pmid)
                for og in ocurrencias_repetidas_no_analizadas:
                    genes = alias_gen[og]
                    reemplazar_por = []
                    for g in genes: # si el alias se mapea a varios genes, tratar de elegir los etiquetados
                        if g in genes_etiquetados_no_encontrados:
                            reemplazar_por.append(g)
                            genes_etiquetados_encontrados.add(g)
                    if reemplazar_por:
                        contents = replace(contents, og, reemplazar_por, pmid)
                        logging.info("Pmid %s: La ocurrencia %s (con embedding) tiene varios nombres reales de genes (%s), se eligió: %s por estar etiquetados",
                                     pmid, og, str(genes), str(reemplazar_por))

    genes_etiquetados_no_encontrados = set(genes_etiquetados) - genes_etiquetados_encontrados
    if genes_etiquetados_no_encontrados:
        logging.warning("Pmid %s: De los genes etiquetados, no se encontró: %s", pmid, str(genes_etiquetados_no_encontrados))
    
    
    
    # REEMPLAZO PARA DROGAS -----------------------------------------------------------

    drogas_etiquetadas = list(etiquetas_drogas[pmid])
    drogas_etiquetadas_encontradas = set()
    
    for og in ocurrencias_drogas_se_sr[pmid]: # analizo sin embedding - sin repetición
        drogas = alias_droga[og]
        if len(drogas) != 1:
            logging.error("Usando SIN REPETICIONES se encontró un alias de droga repetido!!: %s", og)
            logging.error("Diccionario: %s", str(drogas))
        droga = list(drogas)[0]
        contents = replace(contents, og, droga, pmid)
        if droga in drogas_etiquetadas:
            drogas_etiquetadas_encontradas.add(droga)
    
    if set(drogas_etiquetadas) != drogas_etiquetadas_encontradas:
        # analizo sin embedding - con repetición
        ocurrencias_repetidas_no_analizadas = set(ocurrencias_drogas_se_cr[pmid]) - set(ocurrencias_drogas_se_sr[pmid])
        for og in ocurrencias_repetidas_no_analizadas:
            drogas = alias_droga[og]
            reemplazar_por = []
            for d in drogas: # si el alias se mapea a varias drogas, tratar de elegir las etiquetadas
                if d in drogas_etiquetadas:
                    reemplazar_por.append(d)
                    drogas_etiquetadas_encontradas.add(d)
            if not reemplazar_por:
                # ninguna etiquetada, elijo todas
                reemplazar_por = drogas
                logging.warning("Pmid %s: La ocurrencia %s tiene varios nombres reales de drogas (%s), se eligieron todos: %s",
                                pmid, og, str(drogas), str(reemplazar_por))
            else:
                logging.info("Pmid %s: La ocurrencia %s tiene varios nombres reales de drogas (%s), se eligió: %s por estar etiquetadas",
                            pmid, og, str(drogas), str(reemplazar_por))
            
            contents = replace(contents, og, reemplazar_por, pmid)

            if set(drogas_etiquetadas) != drogas_etiquetadas_encontradas:
                # analizo con embedding - con repetición
                drogas_etiquetadas_no_encontradas = set(drogas_etiquetadas) - drogas_etiquetadas_encontradas
                ocurrencias_repetidas_no_analizadas = set(ocurrencias_drogas_todas[pmid]) - set(ocurrencias_drogas_se_cr[pmid])
                logging.info("Pmid %s: Analizando droga ahora con embedding - con repetición...", pmid)
                for og in ocurrencias_repetidas_no_analizadas:
                    drogas = alias_droga[og]
                    reemplazar_por = []
                    for g in drogas: # si el alias se mapea a varias drogas, tratar de elegir las etiquetadas
                        if g in drogas_etiquetadas_no_encontradas:
                            reemplazar_por.append(g)
                            drogas_etiquetadas_encontradas.add(g)
                    if reemplazar_por:
                        contents = replace(contents, og, reemplazar_por, pmid)
                        logging.info("Pmid %s: La ocurrencia %s (con embedding) tiene varios nombres reales de drogas (%s), se eligió: %s por estar etiquetados",
                                     pmid, og, str(drogas), str(reemplazar_por))

    drogas_etiquetadas_no_encontradas = set(drogas_etiquetadas) - drogas_etiquetadas_encontradas
    if drogas_etiquetadas_no_encontradas:
        logging.warning("Pmid %s: De las drogas etiquetadas, no se encontró: %s", pmid, str(drogas_etiquetadas_no_encontradas))
    
    if todisk:
        with open(output_dir / (pmid + ".txt"), "w") as f:
            f.write(contents)
    else:
        return contents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
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

    dir_pfc_dgidb = pathlib.Path(dir_pfc_dgidb)
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

    i = 1; tot = len(publicaciones_dict)
    for pmid, contents in publicaciones_dict.items():
        print("Procesando {}/{}".format(i, tot))
        reemplazar_bme(pmid, contents, output_dir)
        i += 1
