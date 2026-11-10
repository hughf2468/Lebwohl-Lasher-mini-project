import numpy as np
from numba import njit

#importing all test scripts.
import LebwohlLasher as orig
import cython_serial.LebwohlLasher_cython_funcs as cs
import cython_omp.LebwohlLasher_cython_funcs_omp as comp
import numba_tests.LebwohlLasher_numba as nb
import numba_tests.LebwohlLasher_numba_parallel as nbp
import numpy_vectorization.LebwohlLasher_np as np_funcs



#Importing libraries for running MPI separately.
import glob, os, subprocess, sys, tempfile, pathlib
mypath=pathlib.Path(__file__).resolve().parent
mpi_code = mypath / "mpi4py"/ "LebwohlLasher_np_mpi.py"

#As the structures of the different code versions are so different, we con't compare the outputs of MC_step directly.
#We can however compare long term equilibrium averages of energy, order and ratio.
#Use a small grid for speed.
nmax = 16
nsteps = 300
#Give it time to reach equilibrium.
burn_in_time = 100
#Ensemble over a number of runs.
ensemble_runs = 10
Ts_vals = [0.5,1.0,1.5]



#Defining versions of the code for testing.
code_versions = {"Original": orig.MC_step, "Numba": nb.MC_step, "Numba Parallel": nbp.MC_step, "Cython": cs.cMC_step, "Cython Parallel": comp.cMC_step,
                 "Chequerboard+NumPy": np_funcs.MC_step}

def ensemble_avg(mcstep, Ts):
    runs = []
    for i in range(ensemble_runs):
        #Random initial lattice as before.
        arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
        # Create lists to store energy, acceptance ratio and order parameter
        energy = np.zeros(nsteps-burn_in_time,dtype=np.float64)
        ratio = np.zeros(nsteps-burn_in_time,dtype=np.float64)
        order = np.zeros(nsteps-burn_in_time,dtype=np.float64)
        for it in range(nsteps):
            mcout = mcstep(arr, Ts, nmax)
            curr_index = it-burn_in_time
            if curr_index>=0: 
                ratio[curr_index] = mcout
                energy[curr_index] = orig.all_energy(arr, nmax)
                order[curr_index] = orig.get_order(arr, nmax)
        runs.append((np.mean(ratio), np.mean(energy), np.mean(order)))
        
    runs = np.array(runs)
    return runs.mean(axis=0), runs.std(axis=0,ddof=1)/np.sqrt(ensemble_runs)

#The MPI code is tricky in import, so running the entire program as a subprocess.
def ensemble_mpi(Ts):
    runs=[]
    for i in range(ensemble_runs):
        #create a temporary directory for the outputs.
        with tempfile.TemporaryDirectory() as tempdir:
            #Running as a subprocess on 4 ranks.
            subprocess.run(["mpirun", "-n", "4", sys.executable, mpi_code, str(nsteps), str(nmax), str(Ts), "0"], cwd=tempdir, check=True, capture_output=True)
            #Capturing the output file (there is only one)
            output_name=glob.glob(os.path.join(tempdir, "LL-Output-*.txt"))[0]
            #Loading the output data, skipping comments that start with "#"
            output_data=np.loadtxt(output_name, comments="#")
        post_burn_in=output_data[burn_in_time+1:]
        runs.append((np.mean(post_burn_in[:,1]), np.mean(post_burn_in[:,2]), np.mean(post_burn_in[:,3])))
    runs = np.array(runs)
    return runs.mean(axis=0), runs.std(axis=0,ddof=1)/np.sqrt(ensemble_runs)




if __name__=="__main__":
   for Ts in Ts_vals:
        print("T = "+str(Ts))
        #Create variables for original means and standard deviations.
        orig_mu = orig_sigma = None
        for name, mcstep in code_versions.items():
            mu, sigma = ensemble_avg(mcstep, Ts)
            if orig_mu is None:
                orig_mu = mu
                orig_sigma = sigma
            else:
                print("For "+name)
                print("Z scores for Ratio, Energy, Order")
                z = (mu-orig_mu)/(np.sqrt(sigma**2+orig_sigma**2))
                print(z)
        #MPI version separately
        print("The MPI output may take some time as the MPI environment must be configured for each run separately.")
        mu, sigma = ensemble_mpi(Ts)
        print("For mpi4py")
        print("Z scores for Ratio, Energy, Order")
        print((mu-orig_mu)/(np.sqrt(sigma**2+orig_sigma**2)))
