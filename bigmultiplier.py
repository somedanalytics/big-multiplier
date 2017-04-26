#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Can Aykul, Orcun Gumus
#

"""A function for multiply 2 big same size matrix"""

import theano
from theano import tensor, function
from scipy import sparse
import numpy as np

# Tensor definitions

x0 = tensor.matrix(name='x0', dtype='float32')
x1 = tensor.matrix(name='x1', dtype='float32')
y = theano.tensor.dot(x0, x1)
multiply = function([x0, x1], y)


def bigmultiplier(A, B, WIDTH=1000):
    """
    A function for multiply 2 big same size matrix
    :param WIDTH: width of blocks
    :param A: The first matrix to be multiplied
    :param A: The second matrix to be multiplied
    :type A: sparse.csr_matrix
    :type B: sparse.csr_matrix
    :return: sparse.csr_matrix
    """
    result_matrix = sparse.lil_matrix(A.shape, dtype=np.float32)

    def num_blocks(rows_per_matrix, size_tuple):
        if (size_tuple[0] // rows_per_matrix) == size_tuple[0] / rows_per_matrix:
            return size_tuple[0] // rows_per_matrix
        else:
            return (size_tuple[0] // rows_per_matrix) + 1

    rows_per_matrix = WIDTH if A.shape[0] > WIDTH else A.shape[0]
    num_blocks = num_blocks(rows_per_matrix=rows_per_matrix, size_tuple=A.shape)

    for row_block in range(0, num_blocks):

        for column_block in range(0, num_blocks):
            start_row = row_block * rows_per_matrix
            end_row = (start_row + rows_per_matrix) if (start_row + rows_per_matrix) <= A.shape[0] else A.shape[0]
            start_col = column_block * rows_per_matrix
            end_col = (start_col + rows_per_matrix) if (start_col + rows_per_matrix) <= A.shape[0] else A.shape[0]

            x0_ = A[start_row:end_row, :].toarray()
            x1_ = B[:, start_col:end_col].toarray()

            val = multiply(x0=x0_, x1=x1_)
            result_matrix_row_start = row_block * rows_per_matrix
            result_matrix_col_start = column_block * rows_per_matrix
            result_matrix[  result_matrix_row_start: result_matrix_row_start + rows_per_matrix,
                            result_matrix_col_start: result_matrix_col_start + rows_per_matrix] = val

    return result_matrix.tocsr()

