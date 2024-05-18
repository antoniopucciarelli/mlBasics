import numpy as np 

def getDim(matrix):
    '''
    This function checks matrix dimension
    '''

    nrow = len(matrix)
    ncol = len(matrix[0])

    return nrow, ncol

def getPos(matrix, nrow, ncol):
    # setting un data
    rowIndex = []
    colIndex = []
    val      = []
    
    for ii in range(nrow):
        for jj in range(ncol):
            value = matrix[ii][jj]
            if value != 0:
                rowIndex.append(ii)
                colIndex.append(jj)
                val.append(value)

    return rowIndex, colIndex, val

def initMatrix(row, col):
    matrix_c = np.zeros(shape=(row, col))
    
    return matrix_c
        
def sparse_matrix_multiplication(matrix_a, matrix_b):
    
    nrow_a, ncol_a = getDim(matrix_a)
    nrow_b, ncol_b = getDim(matrix_b)

    if ncol_a != nrow_b:
        return [[]]
    else:
        rowIndex_a, colIndex_a, val_a = getPos(matrix_a, nrow_a, ncol_a)
        rowIndex_b, colIndex_b, val_b = getPos(matrix_b, nrow_b, ncol_b)

    len_a = len(rowIndex_a)
    len_b = len(rowIndex_b)
            
    matrix_c = initMatrix(nrow_a, ncol_b)
    
    if len_a == 0 or len_b == 0:
        pass
    else:    
        for ii in range(len_a):
            rowID = rowIndex_a[ii]
            
            for jj in range(len_b):
                colID = colIndex_b[jj]

                if colIndex_a[ii] == rowIndex_b[jj]:
                    value = val_a[ii] * val_b[jj]

                    matrix_c[rowID][colID] = matrix_c[rowID][colID] + value
                    
    matrix_c = matrix_c.tolist()
    
    return matrix_c
