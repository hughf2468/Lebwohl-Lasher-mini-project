#cython: profile=True
from libc.math cimport cos
import numpy as np

# cpdef defines a pure c function for maximum speed (as cone_energy is called often).
#Can be called from python (unlike cdef).
# Arr is a typed memoryview.
cpdef inline double cone_energy(double[:,::1] arr, Py_ssize_t ix, Py_ssize_t iy, Py_ssize_t nmax):
    #Coordinates of the neightbours. 
    #Py_ssize_t is the datatype numpy uses for indices and sizes.
    cdef double en = 0.0
    cdef Py_ssize_t ixp = (ix+1)%nmax
    cdef Py_ssize_t ixm = (ix-1)%nmax
    cdef Py_ssize_t iyp = (iy+1)%nmax
    cdef Py_ssize_t iym = (iy-1)%nmax

    cdef double ang, cosresult
    ang = arr[ix, iy] - arr[ixp, iy]
    cosresult = cos(ang)
    en += 0.5 * (1.0 - 3.0*cosresult*cosresult)
    
    ang = arr[ix, iy] - arr[ixm, iy]
    cosresult = cos(ang)
    en += 0.5 * (1.0 - 3.0*cosresult*cosresult)
    
    ang = arr[ix, iy] - arr[ix, iyp]
    cosresult = cos(ang)
    en += 0.5 * (1.0 - 3.0*cosresult*cosresult)

    ang = arr[ix, iy] - arr[ix, iym]
    cosresult = cos(ang)
    en += 0.5 * (1.0 - 3.0*cosresult*cosresult)
    
    return en

def cget_order(arr, Py_ssize_t nmax):
    cdef Py_ssize_t i,j,a,b
    #Using a new variable for the order parameter so that Qab doesn't have to be loaded from ram for every i,j. 
    cdef double Qabab
    cdef double[:, ::1] Qab = np.zeros((3, 3))
    cdef double[:, ::1] delta = np.eye(3, 3)


    cdef double[:, :, ::1] lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)

    for a in range(3):
        for b in range(3):
            Qabab = 0
            for i in range(nmax):
                for j in range(nmax):
                    Qabab += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
            Qab[a, b] = Qabab
    #Converting from typed memoryview to numpy array.
    Qab_np = np.asarray(Qab) / (2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab_np)
    return eigenvalues.max()
    
    