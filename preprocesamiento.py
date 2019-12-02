from datos_preprocesamiento import cargar_publicaciones, cargar_abstracts, cargar_ocurrencias, cargar_aliases_dict, cargar_etiquetas_dict, cargar_pmids
import pathlib
import sys
import argparse
import logging


def reemplazar_bme(contents, ocurrencias):
    pass



if __name__ == "__main__":
    logging.getLogger("datos_preprocesamiento").setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("lista_pmid", help="Archivo con el listado de PMID para trabajar")
    parser.add_argument("dir_txt_publicaciones", help="Directorio con los TXT de los papers")
    parser.add_argument("ruta_abstracts", help="Archivo con los abstracts/títulos/palabras clave")
    parser.add_argument("dir_pfc_dgidb", help="Directorio PFC_DGIdb para obtener archivos con datos")
    args = parser.parse_args()

    print("Cargando todo...")

    abstracts_dict = cargar_abstracts(args.ruta_abstracts)
    publicaciones_dict = cargar_publicaciones(args.dir_txt_publicaciones, abstracts_dict,
                                              cargar_pmids(args.lista_pmid))
    
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

    # for pmid, contents in publicaciones_dict.items():
        # reemplazar_bme(contents, ocurrencias_dict[pmid])

    print("Listo.")