#cython: profile=True
from libc.math cimport cos

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
