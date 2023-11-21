# SpMM using CSR format
# First convert a given sparse matrix to CSR then apply custom dense X CSR matrix multiply


import numpy as np
from typing import Tuple
import scipy.sparse as sp

class CSRMatrix:

    def __init__(self, rowptr: np.ndarray, cols: np.ndarray,
                 values: np.ndarray, shape: Tuple[int, int]) -> None:

        # Row index/pointer
        self.rowptr = rowptr
        # Column index
        self.cols = cols
        # Non-zero values
        self.values = values
        
        self.shape = shape
        self.dtype = self.values.dtype


def convert_dense_to_csr(mat):
    shape = mat.shape
    assert len(shape) == 2
    num_rows = shape[0]
    num_cols = shape[1]

    # The size of rowptr array is num_rows + 1
    rowptr = np.zeros(num_rows + 1).astype(np.int32)
    cols = []
    values = []
    
    for i in range(num_rows):
        for j in range(num_cols):
            if mat[i][j] != 0:
                values.append(mat[i][j])
                cols.append(j)
            rowptr[i + 1] = len(cols)
    cols = np.array(cols).astype(np.int32)
    values = np.array(values).astype(mat.dtype)
    csr_matrix = CSRMatrix(rowptr=rowptr,
                        cols=cols,
                        values=values,
                        shape=shape)
    return csr_matrix



def main():
    # Create a 5x5 sparse matrix with density 0.5
    sparse_matrix = (sp.random(5, 5, density=0.5).toarray()*5).astype(int)
    print(sparse_matrix)
    sparse_csr = convert_dense_to_csr(sparse_matrix)
    
    print("CSR Encoded Sparse Matrix")
    print("rowptr: {}".format(sparse_csr.rowptr))
    print("col index: {}".format(sparse_csr.cols))
    print("Values: {}".format(sparse_csr.values))
   
    # Dense 5 x 5 matrix 
    dense_matrix = np.random.randint(1, 10, (5, 5))
    print(dense_matrix)
    res = np.zeros((5,5))

    # matrix order, assumes both matrices are square
    n = len(dense_matrix)

    # res = dense X csr
    csr_row = 0 # Current row in CSR matrix
    for i in range(n):
        start, end = sparse_csr.rowptr[i], sparse_csr.rowptr[i + 1]
        for j in range(start, end):
            col, csr_value = sparse_csr.cols[j], sparse_csr.values[j]
            print(col)
            print(csr_value)
            for k in range(n):
                dense_value = dense_matrix[k][csr_row]
                res[k][col] += csr_value * dense_value
                print(res)
        csr_row += 1

    print(res)
    
    #Check if (dense X csr) equals (dense X dense-representation-sparse-matrix)
    print(np.array_equal(np.matmul(dense_matrix,sparse_matrix), res))
    
if __name__ == '__main__':
  main()
