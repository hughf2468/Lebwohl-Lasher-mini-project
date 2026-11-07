from LebwohlLasher import one_energy, all_energy, get_order, MC_step
from LebwohlLasher_cython_funcs import cone_energy, cget_order, cMC_step, call_energy
import numpy as np

def lattice_gen(nmax):
    return np.random.random_sample((nmax,nmax))*2.0*np.pi

nmax = 50
arr = lattice_gen(nmax)

def test_one_energy():
    ix = np.random.randint(0, nmax)
    iy = np.random.randint(0, nmax)
    en_orig = one_energy(arr, ix, iy, nmax)
    assert np.isclose(cone_energy(arr, ix, iy, nmax), en_orig)


def test_all_energy():
    enall_orig = all_energy(arr, nmax)
    assert np.isclose(call_energy(arr, nmax),enall_orig)


def test_get_order():
    op_orig = get_order(arr, nmax)
    assert np.isclose(cget_order(arr, nmax), op_orig)

