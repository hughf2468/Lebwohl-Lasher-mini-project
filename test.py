from LebwohlLasher import one_energy, all_energy, get_order, MC_step
from LebwohlLasher_cython_funcs import cone_energy, cget_order, cMC_step, call_energy
import numpy as np

def test_one_energy(arr, nmax):
    ix = np.random.randint(0, nmax)
    iy = np.random.randint(0, nmax)
    en_orig = one_energy(arr, ix, iy, nmax)
    if np.isclose(cone_energy(arr, ix, iy, nmax), en_orig):
        print("Cython one_energy: Pass")
    else:
        print("Cython one_energy: Fail")

def test_all_energy(arr, nmax):
    enall_orig = all_energy(arr, nmax)
    if np.isclose(call_energy(arr, nmax),enall_orig):
        print("Cython all_energy: Pass")
    else:
        print("Cython all_energy: Fail")

def test_get_order(arr, nmax):
    op_orig = get_order(arr, nmax)
    if np.isclose(cget_order(arr, nmax), op_orig):
        print("Cython get_order: Pass")
    else:
        print("Cython get_order: Fail")


if __name__ == "__main__":
    nmax = 50
    #np.random.seed(29)
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    test_one_energy(arr, nmax)
    test_all_energy(arr, nmax)
    test_get_order(arr, nmax)