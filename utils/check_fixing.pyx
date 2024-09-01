import numpy as np
from cython.parallel import prange

cimport numpy as cnp
cimport cython
from cython.view cimport array as cvarray

#DTYPE = np.float
#ctypedef cnp.float_t DTYPE_t


cdef float EPI=1e-4

@cython.boundscheck(False)
@cython.wraparound(False)
def check_fixing_cy(cnp.ndarray[float, ndim=2] instance, cnp.ndarray[float, ndim=2] b_tmp, int idx):
    assert instance.shape[0] - 1 == b_tmp.shape[0]
    cdef int nrow = b_tmp.shape[0]
    cdef int ncol = instance.shape[1]
    cdef bint label = False
    cdef int i, j
    cdef int cnt
    cdef float[:, :] ins_view = instance
    cdef float[:, :] bt_view = b_tmp
    for i in range(1, nrow+1):
        #print(bt_view[i-1, 0])
        if bt_view[i-1, 0] <= 1-EPI:
            continue
        cnt = 0
        for j in range(idx, ncol):
            if ins_view[i,j] >= EPI:
                if abs(ins_view[i,j] - 1) >= EPI:
                    raise ValueError("instance fraction!")
                cnt += 1
        if cnt == 1 and abs(ins_view[i,idx] - 1) <= EPI:
            label = True
    return label
