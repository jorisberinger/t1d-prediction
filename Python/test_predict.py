from unittest import TestCase


class TestPredict(TestCase):
    def test_iob(self):
        from Python.predict import iob
        self.assertEqual(iob(-1, 1), 100)
        self.assertEqual(iob(61, 1), 0)
        self.assertAlmostEqual(iob(12, 3), 98.75, 1)
        self.assertAlmostEqual(iob(56, 4), 81.70, 1)
        self.assertAlmostEqual(iob(33, 5), 95.53, 1)
        self.assertAlmostEqual(iob(270, 6), 17.13, 1)

    def test_intIOB(self):
        from Python.predict import intIOB
        self.assertAlmostEqual(intIOB(10, 20, 1, 5), 973.33, 1)
        self.assertAlmostEqual(intIOB(40, 90, 2, 9), 4866.66, 1)
        self.assertAlmostEqual(intIOB(0, 90, 1, 0), 8760, 1)
        self.assertAlmostEqual(intIOB(40, 90, 3, 70), 4774.41, 1)
        self.assertAlmostEqual(intIOB(0, 90, 1, 20), 8760.0, 1)

    def test_cob(self):
        from Python.predict import cob
        self.assertAlmostEqual(cob(0,1), 0, 1)
        self.assertAlmostEqual(cob(3,-1), 1, 1)
        self.assertAlmostEqual(cob(3,5), 0.68, 1)
        self.assertAlmostEqual(cob(10,50), 0.08, 1)
        self.assertAlmostEqual(cob(10, 15), 0.77, 1)

    def test_deltatempBGI(self):
        from Python.predict import deltatempBGI
        self.assertAlmostEqual(deltatempBGI(10, 4, 2, 3, 30, 60), -6.40, 1)
        self.assertAlmostEqual(deltatempBGI(-3, 1, 2, 3, 30, 60), -1.60, 1)
        self.assertAlmostEqual(deltatempBGI(30, 1, 2, 6, 10, 60), -2.73, 1)
        self.assertAlmostEqual(deltatempBGI(45, 6, 5, 6, 10, 300), -239.61, 1)

    def test_deltaBGC(self):
        from Python.predict import deltaBGC
        self.assertAlmostEqual(deltaBGC(10, 1, 3, 10, 1), 3.33, 1)
        self.assertAlmostEqual(deltaBGC(-5, 1, 3, 10, 1), 0, 1)
        self.assertAlmostEqual(deltaBGC(5, 2, 3, 16, 1), 10.66, 1)
        self.assertAlmostEqual(deltaBGC(30, 5, 9, 3, 5), 1.66, 1)

    def test_deltaBGI(self):
        from Python.predict import deltaBGI
        self.assertAlmostEqual(deltaBGI(30, 5, 9, 3), -4.36, 1)
        self.assertAlmostEqual(deltaBGI(-5, 6, 9, 3), 0, 1)
        self.assertAlmostEqual(deltaBGI(60, 10, 3, 4), -6.09, 1)
        self.assertAlmostEqual(deltaBGI(20, 1, 5, 2), -0.0, 1)

    def test_deltaBG(self):
        from Python.predict import deltaBG
        self.assertAlmostEqual(deltaBG(3, 5, 3, 1, 3, 5, 2), 1.6666666666666667, 1)
        self.assertAlmostEqual(deltaBG(20, 1, 5, 2, 5, 6, 5), 0.28, 1)
        self.assertAlmostEqual(deltaBG(30, 2, 5, 10, 3, 10, 2), 4.0, 1)
