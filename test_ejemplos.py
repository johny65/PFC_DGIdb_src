"""
Testea la carga de ejemplos (generación de matrices).
"""
import unittest
import numpy as np
from redes_neuronales_preprocesamiento import *


class Test(unittest.TestCase):

    def test_carga(self):
        (xe, ye), (xt, yt) = cargar_ejemplos("tests/test_etiquetas_2", "tests",
                                             "tests/test_int", porcentaje_test=0.0,
                                             embeddings_file="tests/test_emb",
                                             max_longitud=15, randomize=False)
        self.assertEqual(0, len(xt))
        self.assertEqual(0, len(yt))
        self.assertEqual(2, len(xe))
        self.assertEqual(2, len(ye))
        
        # nota: la separación en palabras deja afuera las de 3 o menos caracteres
        r0 = np.asarray([
            [0.0, 0.0, 0.0, 0.0, 0.0], # PADDING
            [0.0, 0.0, 0.0, 0.0, 0.0], # PADDING
            [0.0, 0.0, 0.0, 0.0, 0.0], # PADDING
            [0.0, 0.0, 0.0, 0.0, 0.0], # PADDING
            [0.0, 0.0, 0.0, 0.0, 0.0], # PADDING
            [0.0, 0.0, 0.0, 0.0, 0.0], # PADDING
            [5.1, 5.2, 5.3, 0.0, 0.0], # sita
            [0.0, 0.0, 0.0, 1.0, 0.0], # gen g1 (de interés)
            [1.1, 1.2, 1.3, 0.0, 0.0], # commodo
            [2.1, 2.2, 2.3, 0.0, 0.0], # cillum
            [0.7, 0.8, 0.9, 0.0, 0.0], # esse
            [0.0, 0.0, 0.0, 0.0, 0.0], # d2
            [3.1, 3.2, 3.3, 0.0, 0.0], # aliqua
            [1.0, 2.0, 3.0, 0.0, 0.0], # pariatur
            [0.0, 0.0, 0.0, 0.0, 1.0]  # droga d1 (de interés)
            ])

        print(xe[0].shape, r0.shape)
        self.assertEqual(xe[0].shape, r0.shape)
        self.assertTrue((xe[0] == r0).all())

        r1 = np.asarray([
            [0.0, 0.0, 0.0, 0.0, 0.0], # PADDING
            [0.0, 0.0, 0.0, 0.0, 0.0], # PADDING
            [0.0, 0.0, 0.0, 0.0, 0.0], # PADDING
            [0.0, 0.0, 0.0, 0.0, 0.0], # PADDING
            [0.0, 0.0, 0.0, 0.0, 0.0], # PADDING
            [0.0, 0.0, 0.0, 0.0, 0.0], # PADDING
            [5.1, 5.2, 5.3, 0.0, 0.0], # sita
            [0.0, 0.0, 0.0, 1.0, 0.0], # gen g1 (de interés)
            [1.1, 1.2, 1.3, 0.0, 0.0], # commodo
            [2.1, 2.2, 2.3, 0.0, 0.0], # cillum
            [0.7, 0.8, 0.9, 0.0, 0.0], # esse
            [0.0, 0.0, 0.0, 0.0, 1.0], # droga d2 (de interés)
            [3.1, 3.2, 3.3, 0.0, 0.0], # aliqua
            [1.0, 2.0, 3.0, 0.0, 0.0], # pariatur
            [0.0, 0.0, 0.0, 0.0, 0.0]  # d1
            ])

        print(xe[1].shape, r1.shape)
        self.assertEqual(xe[1].shape, r1.shape)
        self.assertTrue((xe[1] == r1).all())


    def test_carga_transpuesta(self):
        (xe, ye), (xt, yt) = cargar_ejemplos("tests/test_etiquetas_2", "tests",
                                             "tests/test_int", porcentaje_test=0.0,
                                             embeddings_file="tests/test_emb",
                                             max_longitud=15, randomize=False)
        x = xe[0].transpose()
        r0 = np.asarray([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.1, 0.0, 1.1, 2.1, 0.7, 0.0, 3.1, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.2, 0.0, 1.2, 2.2, 0.8, 0.0, 3.2, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.3, 0.0, 1.3, 2.3, 0.9, 0.0, 3.3, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ])
        print(x.shape, r0.shape)
        self.assertEqual(x.shape, r0.shape)
        self.assertTrue((x == r0).all())

        x = xe[1].transpose()
        r1 = np.asarray([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.1, 0.0, 1.1, 2.1, 0.7, 0.0, 3.1, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.2, 0.0, 1.2, 2.2, 0.8, 0.0, 3.2, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.3, 0.0, 1.3, 2.3, 0.9, 0.0, 3.3, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            ])
        print(x.shape, r1.shape)
        self.assertEqual(x.shape, r1.shape)
        self.assertTrue((x == r1).all())


if __name__ == "__main__":
    unittest.main()
