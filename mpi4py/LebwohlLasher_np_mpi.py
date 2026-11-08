"""
Basic Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#Import and initialise MPI
from mpi4py import MPI
#Find out my task and what my ID is.
comm = MPI.COMM_WORLD
taskid = comm.Get_rank()
numtasks = comm.Get_size()
#numworkers = numtasks-1

#=======================================================================
# def initdat(nmax):
#     """
#     Arguments:
#       nmax (int) = size of lattice to create (nmax,nmax).
#     Description:
#       Function to create and initialise the main data array that holds
#       the lattice.  Will return a square lattice (size nmax x nmax)
# 	  initialised with random orientations in the range [0,2pi].
# 	Returns:
# 	  arr (float(nmax,nmax)) = array to hold lattice.
#     """
#     arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
#     return arr
#=======================================================================
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================
def one_energy(arr,ix,iy,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en

#=======================================================================
#Must be separated out so it can be called by MC_step.
def energy_arr(arr):
    #enarr holds the array of all of the energies in the lattice.
    enarr = 0.0
    #Iterating over axis and neighbour each side.
    for xydirection in (0,1):
        for neighbourplusminus in (1,-1):
            #np.roll will give a copy of the shifted lattice, with wraparound.
            arr_shift = np.roll(arr, neighbourplusminus, xydirection)
            #Calculating differences in the shifted lattice.
            ang = arr-arr_shift
            #Calculate energies for the shifted lattice.
            enarr += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return enarr

def all_energy(arr,nmax):
    halo(arr)
    return comm.allreduce(energy_arr(arr)[1:-1].sum())



# def all_energy(arr,nmax):
#     """
#     Arguments:
# 	  arr (float(nmax,nmax)) = array that contains lattice data;
#       nmax (int) = side length of square lattice.
#     Description:
#       Function to compute the energy of the entire lattice. Output
#       is in reduced units (U/epsilon).
# 	Returns:
# 	  enall (float) = reduced energy of lattice.
#     """
#     enall = 0.0
#     for i in range(nmax):
#         for j in range(nmax):
#             enall += one_energy(arr,i,j,nmax)
#     return enall
#=======================================================================
# def get_order(arr,nmax):
#     """
#     Arguments:
# 	  arr (float(nmax,nmax)) = array that contains lattice data;
#       nmax (int) = side length of square lattice.
#     Description:
#       Function to calculate the order parameter of a lattice
#       using the Q tensor approach, as in equation (3) of the
#       project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
# 	Returns:
# 	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
#     """
#     Qab = np.zeros((3,3))
#     delta = np.eye(3,3)
#     #
#     # Generate a 3D unit vector for each cell (i,j) and
#     # put it in a (3,i,j) array.
#     #
#     lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
#     for a in range(3):
#         for b in range(3):
#             for i in range(nmax):
#                 for j in range(nmax):
#                     Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
#     Qab = Qab/(2*nmax*nmax)
#     eigenvalues,eigenvectors = np.linalg.eig(Qab)
#     return eigenvalues.max()
def get_order(arr, nmax):
    nohaloarr = arr[1:-1]
    #Generate a 3D unit vector for each cell (i,j) and put it in a (3,i,j) array.
    #as before
    lab = np.vstack((np.cos(nohaloarr),np.sin(nohaloarr),np.zeros_like(nohaloarr))).reshape(3,nohaloarr.shape[0],nmax)
    #Multiply lab by itself elementwise and sum over i and j.
    #axes=([1,2],[1,2]) gives the i and j axes.
    labsum = np.tensordot(lab, lab, axes=([1,2],[1,2]))
    #Reduce labsum over ranks.
    labsum = comm.allreduce(labsum)
    delta = np.eye(3,3)
    Qab = 3*labsum-nmax*nmax*delta
    Qab = Qab/(2*nmax*nmax)
    #Use np.linalg.eigvalsh to avoid calculating eigenvectors.
    return np.linalg.eigvalsh(Qab).max()


#=======================================================================
# def MC_step(arr,Ts,nmax):
#     """
#     Arguments:
# 	  arr (float(nmax,nmax)) = array that contains lattice data;
# 	  Ts (float) = reduced temperature (range 0 to 2);
#       nmax (int) = side length of square lattice.
#     Description:
#       Function to perform one MC step, which consists of an average
#       of 1 attempted change per lattice site.  Working with reduced
#       temperature Ts = kT/epsilon.  Function returns the acceptance
#       ratio for information.  This is the fraction of attempted changes
#       that are successful.  Generally aim to keep this around 0.5 for
#       efficient simulation.
# 	Returns:
# 	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
#     """
#     #
#     # Pre-compute some random numbers.  This is faster than
#     # using lots of individual calls.  "scale" sets the width
#     # of the distribution for the angle changes - increases
#     # with temperature.
#     scale=0.1+Ts
#     accept = 0
#     xran = np.random.randint(0,high=nmax, size=(nmax,nmax))
#     yran = np.random.randint(0,high=nmax, size=(nmax,nmax))
#     aran = np.random.normal(scale=scale, size=(nmax,nmax))
#     for i in range(nmax):
#         for j in range(nmax):
#             ix = xran[i,j]
#             iy = yran[i,j]
#             ang = aran[i,j]
#             en0 = one_energy(arr,ix,iy,nmax)
#             arr[ix,iy] += ang
#             en1 = one_energy(arr,ix,iy,nmax)
#             if en1<=en0:
#                 accept += 1
#             else:
#             # Now apply the Monte Carlo test - compare
#             # exp( -(E_new - E_old) / T* ) >= rand(0,1)
#                 boltz = np.exp( -(en1 - en0) / Ts )

#                 if boltz >= np.random.uniform(0.0,1.0):
#                     accept += 1
#                 else:
#                     arr[ix,iy] -= ang
#     return accept/(nmax*nmax)

#Each task has 2 extra "halo" rows for efficiently exchanging information with neighbouring tasks.
def halo(task_arr):
    #Work out tasks above and below for halo exchange. %numtasks preserves top bottom wraparound.
    task_above = (taskid-1)%numtasks
    task_below = (taskid+1)%numtasks
    #Sending sendbuf my (owned) top row up, also recieving the below rank's top row (to bottom halo).
    comm.Sendrecv(sendbuf=task_arr[1], dest=task_above, recvbuf=task_arr[-1], source=task_below)
    #Doing the same thing the other way.
    comm.Sendrecv(sendbuf=task_arr[-2], dest=task_below, recvbuf=task_arr[0], source=task_above)



    
#This function divides the array like a "chequerboard". It is updated "black" then "white".
#This allows for the whole array to be updates in 2 steps without nearest neighbour conflicts.
def MC_step(task_arr, Ts, nmax, tophalopos):
    #Find the number of non halo rows this task posesses.
    task_rows = task_arr.shape[0]-2
    #Define scale and accept as before.
    scale=0.1+Ts
    accept = 0

    #Create two matrices that are the same dimensions as the task_arr (including halos), containing each elements row and column index respectively.
    row_id, column_id = np.indices((task_rows+2,nmax))

    #Synchronise the row_ids with the full global lattice.
    #This ensures that the chequerboard pattern is synchronised across ranks.
    sync_row_id = row_id-1+tophalopos

    for blackorwhite in (0,1):
        #update neighbours.
        halo(task_arr)
        #use row and column id to create chequerboard pattern on an nmax*nmax array.
        #Even numbers are "black", odd numbers are "white".
        chequerboard = (sync_row_id+column_id)%2
        #Convert the 1s and 0s to trues and falses.
        chequerbool = (chequerboard == blackorwhite)
        #We don't want to update rows we don't own.
        chequerbool[0]=chequerbool[-1]=False
        #compute some random angle changes for every point on the lattice.
        aran = np.random.normal(scale=scale, size=task_arr.shape)
        #Update the lattice, either arr+aran or unchanged depending on current colour.
        new_arr = np.where(chequerbool, task_arr+aran, task_arr)
        #Calculate energy change array for the whole lattice.
        en_change = energy_arr(new_arr) - energy_arr(task_arr)
        #Always accept a decrease in energy.
        dec_en = (en_change<=0)
        #Apply the monte-carlo test on the rest of them.
        #first, generate an array of random numbers between 0 and 1.
        compare_num = np.random.uniform(0.0,1.0,(task_arr.shape))
        boltz=np.exp(-en_change/Ts)
        approved_in_en = (boltz >= compare_num)
        #Transitions are only accepted on the right colour, and if either decrease or approved increase in energy.
        accepted_pos = chequerbool & (dec_en | approved_in_en)
        #Change lattice at accepted positions only.
        task_arr[accepted_pos] = new_arr[accepted_pos]
        accept += accepted_pos.sum()
    #return combined result from all processes.
    return comm.allreduce(accept)/(nmax*nmax)



#=======================================================================
#Function for assembling the full lattice when necessary:
def gather(task_arr, nmax, positions_in_each_task, positions_before_each_task):
    non_halo_arr = np.ascontiguousarray(task_arr[1:-1])
    #only the 0 (master) task recieves the full lattice.
    if taskid==0:
        #Create full array.
        arr=np.empty((nmax, nmax))
        comm.Gatherv(non_halo_arr, [arr, positions_in_each_task, positions_before_each_task, MPI.DOUBLE], root=0)
        return arr
    #the other tasks just send their piece to the root.
    comm.Gatherv(non_halo_arr, None, root=0)
    return None


def main(program, nsteps, nmax, temp, pflag):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    #Validating that the lattice is even*even, required for wraparound values to be different colours on the chequerboard.
    
    if nmax %2:
        if taskid==0:
            print("Lattice size must be even.")
        sys.exit(1)    
    #Also validates that the size of the lattice is greater than the number of threads.
    if  nmax < numtasks:
        if taskid==0:
            print("Lattice size must be greater than number of MPI tasks.")
        sys.exit(1)
    #Assign each task a number of rows.
    #Find the minimum number of rows each task will be assigned, as well as rows that don't fit neatly.
    lowest_num_rows, spare_rows = divmod(nmax, numtasks)
    #Allocate remaining rows on a "first come, first served" basis.
    #Each task computes all other tasks' rows, as this is required for chequerboard sychronisation.
    all_tasks_rows = []
    for i in range(numtasks):
        if i<spare_rows:
            all_tasks_rows.append(lowest_num_rows+1)
        else:
            all_tasks_rows.append(lowest_num_rows)
    #finding this task's num of allocated rows and position of top halo.
    my_rows_count = all_tasks_rows[taskid]
    tophalopos = sum(all_tasks_rows[:taskid])

    #Finding expected numbers and cumulative previous num of numbers for each task (needed for gathering the full lattice).
    positions_in_each_task=[]
    for j in all_tasks_rows:
        positions_in_each_task.append(j*nmax)
    positions_before_each_task = []
    for k in range(numtasks):
        positions_before_each_task.append(sum(positions_in_each_task[:k]))

    # Create and initialise lattice
    #Can't use initdat as it only works with square lattices
    #initialise local lattice with halos
    my_lattice = np.zeros((my_rows_count+2, nmax))
    #fill everywhere except halos with random numbers
    my_lattice[1:-1]=np.random.random_sample((my_rows_count,nmax))*2.0*np.pi
    
    # Plot initial frame of lattice
    init_lattice = gather(my_lattice, nmax, positions_in_each_task, positions_before_each_task)
    if taskid==0:    
        plotdat(init_lattice,pflag,nmax)
    # Create arrays to store energy, acceptance ratio and order parameter
    energy = np.zeros(nsteps+1,dtype=np.float64)
    ratio = np.zeros(nsteps+1,dtype=np.float64)
    order = np.zeros(nsteps+1,dtype=np.float64)
    # Set initial values in arrays
    energy[0] = all_energy(my_lattice,nmax)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(my_lattice,nmax)

    # Begin doing and timing some MC steps.
    #Ensure all tasks are in the same place before timer starts.
    comm.Barrier()

    initial = MPI.Wtime()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(my_lattice,temp,nmax, tophalopos)
        energy[it] = all_energy(my_lattice,nmax)
        order[it] = get_order(my_lattice,nmax)
    #Ensure all tasks are in the same place before timer ends.
    comm.Barrier()
    final = MPI.Wtime()
    runtime = final-initial
    
    # Final outputs
    final_lattice = gather(my_lattice, nmax, positions_in_each_task, positions_before_each_task)
    if taskid ==0:
        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
        # Plot final frame of lattice and generate output file
        savedat(final_lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
        plotdat(final_lattice,pflag,nmax)
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================
