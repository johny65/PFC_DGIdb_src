"""
Testea el algoritmo de búsqueda de ocurrencias de genes/drogas en los artículos.
"""
import unittest
import os
from datos_preprocesamiento import *


class Test(unittest.TestCase):

    def test_ocurrencias(self):
        self.generar_ocurrencias()

        self.assertEqual(2, len(self.g_se_sr))
        self.assertEqual({"1234", "holagen"}, set(self.g_se_sr))

        self.assertEqual(2, len(self.g_se_cr))
        self.assertEqual({"1234", "holagen"}, set(self.g_se_cr))

        self.assertEqual(3, len(self.g_ce_cr))
        self.assertEqual({"1234", "holagen", "esse"}, set(self.g_ce_cr))

        self.assertEqual(3, len(self.d_se_sr))
        self.assertEqual({"1234", "holadroga", "holaotradroga"}, set(self.d_se_sr))

        self.assertEqual(3, len(self.d_se_cr))
        self.assertEqual({"1234", "holadroga", "holaotradroga"}, set(self.d_se_cr))

        self.assertEqual(4, len(self.d_ce_cr))
        self.assertEqual({"1234", "holadroga", "holaotradroga", "sit"}, set(self.d_ce_cr))


    def generar_ocurrencias(self):
        pmids_etiquetas_completas_csv = "tests/test_pmids"
        pmids_lista = cargar_pmids(pmids_etiquetas_completas_csv)
        self.assertEqual(1, len(pmids_lista))
        print("pmids cargados")

        aliases_gen_ruta = "tests/test_alias_gen"
        aliases_gen_conjunto = cargar_aliases_conjunto(aliases_gen_ruta)
        self.assertEqual(4, len(aliases_gen_conjunto))
        aliases_gen_lista = cargar_aliases_lista(aliases_gen_ruta)
        self.assertEqual(1, len(aliases_gen_lista))
        print("alias gen cargados")

        aliases_droga_ruta = "tests/test_alias_droga"
        aliases_droga_conjunto = cargar_aliases_conjunto(aliases_droga_ruta)
        self.assertEqual(7, len(aliases_droga_conjunto))
        aliases_droga_lista = cargar_aliases_lista(aliases_droga_ruta)
        self.assertEqual(2, len(aliases_droga_lista))
        print("alias droga cargados")

        repeticiones_genes_lista = alias_repetidos(aliases_gen_lista)
        repeticiones_drogas_lista = alias_repetidos(aliases_droga_lista)
        print("repeticiones cargadas")

        embeddings_ruta = "tests/test_emb"
        embeddings_dict = cargar_embeddings(embeddings_ruta)[0] # devuelve 3 elementos
        self.assertEqual(6, len(embeddings_dict))
        print("Embeddings cargados")

        pubs = cargar_publicaciones("tests", {}, pmids_lista)
        self.assertEqual(1, len(pubs))
        print("publicaciones cargadas")

        print("Buscando ocurrencias de genes...")
        g1, g2, g3 = ocurrencias(aliases_gen_conjunto, pubs, embeddings_dict, repeticiones_genes_lista, "g_test", 0)
        self.g_se_sr = g1
        self.g_se_cr = g2
        self.g_ce_cr = g3

        print("Buscando ocurrencias de drogas...")
        d1, d2, d3 = ocurrencias(aliases_droga_conjunto, pubs, embeddings_dict, repeticiones_drogas_lista, "d_test", 0)
        self.d_se_sr = d1
        self.d_se_cr = d2
        self.d_ce_cr = d3
        
        # limpio archivos generados:
        os.remove("g_test_se_sr_0.csv")
        os.remove("g_test_se_cr_0.csv")
        os.remove("g_test_ce_cr_0.csv")
        os.remove("d_test_se_sr_0.csv")
        os.remove("d_test_se_cr_0.csv")
        os.remove("d_test_ce_cr_0.csv")


if __name__ == "__main__":
    unittest.main()
