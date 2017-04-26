#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Can Aykul, Orcun Gumus
import unittest
import numpy

from scipy.sparse import csr_matrix
from bigmultiplier import bigmultiplier


class TestCalculateMethod(unittest.TestCase):

    def test_basic_calculate(self):
        A = csr_matrix([[1, 2], [3, 4]], dtype=numpy.float32)
        A2_calculated = bigmultiplier(A, A).toarray()
        A2_real = numpy.dot(A.toarray(), A.toarray())
        self.assertAlmostEqual(A2_calculated.tolist(), A2_real.tolist(), delta=0.1)

    def test_on_bigger(self):
        A = csr_matrix(numpy.ones((100, 100), dtype=numpy.float32))
        A2_calculated = bigmultiplier(A, A, WIDTH=10).toarray()
        A2_real = numpy.dot(A.toarray(), A.toarray())
        self.assertAlmostEqual(A2_calculated.tolist(), A2_real.tolist(), delta=0.1)

    def test_on_slicing(self):
        A = csr_matrix(numpy.ones((100, 100), dtype=numpy.float32))
        A2_calculated = bigmultiplier(A, A, WIDTH=7).toarray()
        A2_real = numpy.dot(A.toarray(), A.toarray())
        self.assertAlmostEqual(A2_calculated.tolist(), A2_real.tolist(), delta=0.1)


if __name__ == '__main__':
    unittest.main()
