"""
Testea el algoritmo de reemplazos.
"""
import unittest
import os
from datos_preprocesamiento import *
from preprocesamiento import *


class Test(unittest.TestCase):

    def test_carga(self):
        self.assertEqual(1, len(publicaciones_dict))
        self.assertEqual(4, len(alias_gen))
        self.assertEqual(7, len(alias_droga))
        self.assertEqual(1, len(etiquetas_genes))
        self.assertEqual(1, len(next(iter(etiquetas_genes.values()))))
        self.assertEqual(1, len(etiquetas_drogas))
        self.assertEqual(1, len(next(iter(etiquetas_drogas.values()))))

    def test_reemplazo(self):
        pmid, contents = next(iter(publicaciones_dict.items()))
        res = full_reemplazar_bme(pmid, contents, alias_gen, alias_droga, etiquetas_genes, etiquetas_drogas,
                                  ocurrencias_genes_se_sr, ocurrencias_genes_se_cr, ocurrencias_genes_todas,
                                  ocurrencias_drogas_se_sr, ocurrencias_drogas_se_cr, ocurrencias_drogas_todas,
                                  "", False)
        self.assertEqual("ex  xxxg1xxx  ut commodo cillum esse commodo aliqua sit pariatur  xxxd1xxx  ullamco minim,  xxxd2xxx .",
                         res)


if __name__ == "__main__":
    lista_pmid = "tests/test_pmids"
    publicaciones_dict = cargar_publicaciones("tests", {}, cargar_pmids(lista_pmid))
    alias_gen = cargar_aliases_dict("tests/test_alias_gen")
    alias_droga = cargar_aliases_dict("tests/test_alias_droga")
    
    ocurrencias_genes_se_sr = {"1234": ["holagen"]}
    ocurrencias_genes_se_cr = {"1234": ["holagen"]}
    ocurrencias_genes_todas = {"1234": ["holagen", "esse"]}
    ocurrencias_drogas_se_sr = {"1234": ["holadroga", "holaotradroga"]}
    ocurrencias_drogas_se_cr = {"1234": ["holadroga", "holaotradroga"]}
    ocurrencias_drogas_todas = {"1234": ["holadroga", "holaotradroga", "sit"]}
    
    etiquetas_genes, etiquetas_drogas = cargar_etiquetas_dict("tests/test_etiquetas")
    
    unittest.main()
