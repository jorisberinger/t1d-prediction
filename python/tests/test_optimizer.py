from unittest import TestCase

from Classes import PredictionWindow, UserData
import numpy as np
import pandas as pd
class TestOptimizer(TestCase):
    def test_get_time_steps(self):

        ud = UserData(simlength = 2, predictionlength = 60, bginitial = 1, cratio = 1, idur = 1, stats = 1, sensf = 1, inputeeffect = 1)
        pw = PredictionWindow()
        pw.userData = ud

        from optimizer import get_error_time_steps
        ts = get_error_time_steps(pw, 10)
        self.assertEqual(7, len(ts))
        self.assertEqual(60, ts[-1])
        self.assertEqual(0, ts[0])

        pw.userData.simlength = 11
        ts = get_error_time_steps(pw, 60)
        self.assertEqual(11, len(ts))
        self.assertEqual(600, ts[-1])
        self.assertEqual(0, ts[0])

        pw.userData.simlength = 13
        pw.userData.predictionlength = 180
        ts = get_error_time_steps(pw, 15)
        self.assertEqual(41, len(ts))
        self.assertEqual(600, ts[-1])
        self.assertEqual(0, ts[0])


    def test_get_real_values(self):
        ud = UserData(simlength = 13, predictionlength = 180, bginitial = 1, cratio = 1, idur = 1, stats = 1, sensf = 1,
                      inputeeffect = 1)
        pw = PredictionWindow()
        pw.userData = ud
        pw.cgmY = pd.Series(np.array([1] * 601))
        self.assertEqual(601, len(pw.cgmY))

        from optimizer import get_error_time_steps
        ts = get_error_time_steps(pw, 15)

        from optimizer import get_real_values
        rv = get_real_values(pw, ts)
        self.assertEqual(len(ts), len(rv))
        self.assertEqual(len(rv), sum(rv))


    def test_get_cob_matrix(self):
        from optimizer import get_cob_matrix
        cob_matrix = get_cob_matrix(np.array([0, 10, 20, 30, 40, 50, 60]), np.array([0, 30, 60]), [10, 30])
        self.assertIsInstance(cob_matrix, np.matrix)
        self.assertEqual(7 * 2, cob_matrix.shape[0])
        self.assertEqual(3, cob_matrix.shape[1])
        cob_matrix = get_cob_matrix(np.array([0, 10, 20, 30, 40, 50, 60]), np.array([0, 30, 60]), [10, 30, 40, 60])
        self.assertIsInstance(cob_matrix, np.matrix)
        self.assertEqual(7 * 4, cob_matrix.shape[0])
        self.assertEqual(3, cob_matrix.shape[1])
        cob_matrix = get_cob_matrix(np.array([0, 30, 60]), np.array([0, 30, 60]), [10, 30, 40, 60])
        self.assertIsInstance(cob_matrix, np.matrix)
        self.assertEqual(3 * 4, cob_matrix.shape[0])
        self.assertEqual(3, cob_matrix.shape[1])
        cob_matrix = get_cob_matrix(np.array([0, 30, 60]), np.array([0, 15, 30, 45, 60]), [10, 30, 40, 60])
        self.assertIsInstance(cob_matrix, np.matrix)
        self.assertEqual(3 * 4, cob_matrix.shape[0])
        self.assertEqual(5, cob_matrix.shape[1])
        cob_matrix = get_cob_matrix(np.array([0, 30, 60]), np.array([0, 15, 30, 45, 60]), [10])
        self.assertIsInstance(cob_matrix, np.matrix)
        self.assertEqual(3, cob_matrix.shape[0])
        self.assertEqual(5, cob_matrix.shape[1])



