from LebwohlLasher import one_energy, all_energy, get_order, MC_step
from LebwohlLasher_cython_funcs import cone_energy, cget_order, cMC_step, call_energy
import numpy as np
import pytest

def lattice_gen():
    #creating a reproducible random number generator.
    rand = np.random.default_rng(92)
    #creating different starting arrs for testing purposes.
    #completely random angles
    random_arr = rand.random((nmax, nmax))*2.0*np.pi
    #array filled with zeros
    zero_arr = np.zeros((nmax, nmax))
    #array filled with pi/4
    pi4_arr = np.full((nmax,nmax), np.pi/4)
    #array filled with nontrivial angle (1 radian)
    one_arr = np.full((nmax,nmax), 1.0)
    #half angle 0 and half pi/2. Tests wraparound and a firm domain wall.
    half_half_arr = np.zeros((nmax, nmax))
    half_half_arr[nmax//2:, :] = np.pi/2
    #mostly ordered lattice with normally distributed values (sd = 0.2) around a mean of 0.4 rad.
    normal_arr = 0.2*rand.standard_normal((nmax, nmax)) +0.4

    return [random_arr, zero_arr, pi4_arr, one_arr, half_half_arr, normal_arr]

nmax = 50
rtol = 1e-10
atol = 1e-12

arrs = lattice_gen()

@pytest.mark.parametrize("arr", arrs)
def test_one_energy(arr):
    for ix in range(nmax):
        for iy in range(nmax):
            en_orig = one_energy(arr, ix, iy, nmax)
            assert np.isclose(cone_energy(arr, ix, iy, nmax), en_orig, rtol=rtol, atol=atol)

@pytest.mark.parametrize("arr", arrs)
def test_all_energy(arr):
    enall_orig = all_energy(arr, nmax)
    assert np.isclose(call_energy(arr, nmax),enall_orig, rtol=rtol, atol=atol)

@pytest.mark.parametrize("arr", arrs)
def test_get_order(arr):
    op_orig = get_order(arr, nmax)
    assert np.isclose(cget_order(arr, nmax), op_orig, rtol=rtol, atol=atol)

