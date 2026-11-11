#cython: profile=False
from libc.math cimport cos, exp
import numpy as np

# cpdef defines a pure c function for maximum speed (as cone_energy is called often from other iterative functions).
#Can be called from python (unlike cdef).
# Arr is a typed memoryview.
cpdef inline double cone_energy(double[:,::1] arr, Py_ssize_t ix, Py_ssize_t iy, Py_ssize_t nmax) nogil:
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
    #Using a new variable for the order tensor component so that Qab doesn't have to be loaded from ram for every i,j. 
    cdef double Qabab
    cdef double[:, ::1] Qab = np.zeros((3, 3))
    cdef double[:, ::1] delta = np.eye(3, 3)


    cdef double[:, :, ::1] lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)

    with nogil:
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

cpdef inline Py_ssize_t cMC_step_loop(double[:, ::1] arr, double Ts, Py_ssize_t[:, ::1] xran, Py_ssize_t[:, ::1] yran, double[:, ::1] aran, double[:, ::1] ran_arr, Py_ssize_t nmax) nogil:
    cdef Py_ssize_t i, j, ix, iy
    cdef double ang, en0, en1, boltz
    cdef Py_ssize_t accept = 0
    
    for i in range(nmax):
        for j in range(nmax):
            ix = xran[i, j]
            iy = yran[i, j]
            ang = aran[i, j]

            en0 = cone_energy(arr, ix, iy, nmax)
            arr[ix,iy] +=ang

            en1 = cone_energy(arr, ix, iy, nmax)
            if en1<=en0:
                accept += 1
            else:
                boltz = exp( -(en1 - en0) / Ts )

                if boltz >= ran_arr[i,j]:
                    accept += 1
                else:
                    arr[ix,iy] -= ang
    
    return accept

def cMC_step(double[:, ::1] arr, double Ts, Py_ssize_t nmax):
    cdef double scale = 0.1+Ts
    cdef Py_ssize_t accept

    cdef Py_ssize_t[:, ::1] xran = np.random.randint(0, nmax, size=(nmax, nmax), dtype=np.intp)
    cdef Py_ssize_t[:, ::1] yran = np.random.randint(0, nmax, size=(nmax, nmax), dtype=np.intp)
    cdef double[:, ::1] aran = np.random.normal(scale=scale, size=(nmax, nmax))
    #Generates an array of all required random numbers in advance of iterating through the grid.
    #This removes the for loop dependency on numpy and we can remove the gil.
    cdef double[:, ::1] ran_arr = np.random.random_sample((nmax, nmax))
    
    with nogil:
        accept = cMC_step_loop(arr, Ts, xran, yran, aran, ran_arr, nmax)
        
    return accept / <double>(nmax*nmax)

def call_energy(double[:, ::1] arr, Py_ssize_t nmax):
    cdef Py_ssize_t i, j
    cdef double enall=0.0
    with nogil:
        for i in range(nmax):
            for j in range(nmax):
                enall += cone_energy(arr,i,j,nmax)
    return enall