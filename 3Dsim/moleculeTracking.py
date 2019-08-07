import numpy as np

#directory = 'C:/Users/gabri/Desktop/HutzlerSims/Gas-Simulation/3Dsim/Data/'
#file_ext = 'traj017d.dat'

directory="C:/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/ThermalHeavy/F_Cell/"
infile = 'f100_th.dat'
outfile = 'moleculeTracking_f100.dat'

#This is for running on the cluster
CLUSTER = False
VERBOSE = False

NUM_FRAMES = 10


def main():
    global f, numParticles, frameTimes, molPositions

    currentParticle = 0

    prev_vx, prev_vy, prev_vz = 0, 0, 0

    k_frame = 0
    T_K = 0.0

    for i in range(len(f)):

        if VERBOSE:
            print("\nLine {0}, ".format(i))

        if not np.any(f[i]):
            #If we reach the end of a particle's trajectory

            x, y, z = f[i-1,0], f[i-1,1], f[i-1,2]
            r = np.sqrt(x**2+y**2)

            while k_frame < NUM_FRAMES:
                #Fill the rest of the time frames with the particle's last location
                molPositions.update( {T_K : molPositions[T_K] + [ (r, z) ] } )
                k_frame += 1
                T_K = frameTimes[k_frame]

            currentParticle += 1
            print("Particle {} of {}".format(currentParticle, numParticles))
            k_frame, T_K = 0, 0  #Reset clock to beginning of simulation

            prev_vx, prev_vy, prev_vz = 0, 0, 0

        else:
            x, y, z, vx, vy, vz, tim = f[i]
            r = np.sqrt(x**2+y**2)

            while tim >= T_K:

                if tim == T_K:

                    molPositions.update( {T_K : molPositions[T_K] + [ (r, z) ] } )
                    k_frame += 1
                    T_K = frameTimes[k_frame]

                elif tim > T_K:
                    zk, rk = get_position(x1=x, y1=y, z1=z, t1=tim, vx=prev_vx, vy=prev_vy, vz=prev_vz, t0=T_K)

                    molPositions.update( {T_K : molPositions[T_K] + [ (rk, zk) ] } )
                    k_frame += 1
                    T_K = frameTimes[k_frame]

            #Once the recording-frame time has caught up to the particle simulation time,
            #save previous velocities and move on to the next line in the file
            prev_vx, prev_vy, prev_vz = vx, vy, vz


def get_position(x1, y1, z1, t1, vx, vy, vz, t0):
    '''
    Linearly backtrack from (x1, y1, z1, t1) to find
    where the particle was at time t0, assuming constant
    velocity <vx, vy, vz>.
    Return (z0, r0)
    '''
    delta_t = t1-t0
    x0 = x1 - vx*delta_t
    y0 = y1 - vy*delta_t
    z0 = z1 - vz*delta_t

    return z0, np.sqrt(x0**2+y0**2)
    

# =============================================================================
# Taking map molPositions and writing time frames into a file
# =============================================================================
def writeData(filepath):
    '''
    Writes data matrix from molPositions to a file.
    '''
    global frameTimes, molPositions, numParticles

    with open(filepath, 'a+') as tw:
        tw.write('R   Z   TK  \n')

        for t in frameTimes:
            positions = molPositions[t]

            if len(positions) != numParticles:
                raise ValueError("Wrong number of particles in frame t={}. Only {} recorded".format(round(t,3), len(positions)))

            else:
                for (r,z) in positions:
                    tw.write('   '.join(map(str, [round(r,3), round(z,3), round(t,3)] ))+'\n')

    tw.close()





# =============================================================================
# Initializing data matrix, frameTimes array and positions map
# =============================================================================
def initialize():
    '''
    Gets data from file and creates the main required structures:
    1. Array containing particle trajectory data (f)
    2. Array containing times for particle tracking (frameTimes)
    3. Dictionary mapping times to lists of positions (molPositions)
    '''

    global directory, infile, NUM_FRAMES

    print('Getting data from {}...'.format(infile), end='')
    get_data(directory+infile)

    print('Making frames dictionary for {} frames'.format(NUM_FRAMES))
    make_structures()

    print('Done initializing')


def get_data(filepath):
    '''
    Gets particle trajectory data from file.
    The file should be output from parSimArgs with LITE_MODE = False,
    so that the full trajectory is logged
    '''
    global f, numParticles, MAX_TIME

    f = np.loadtxt(filepath, skiprows=1)

    numParticles = 0
    for i in range(len(f)):
        if not np.any(f[i]):
            numParticles += 1 # count number of particles simulated in file

    print(' {} particles'.format(numParticles))

    MAX_TIME = np.max(f[:,6])
    print("Max time is {}".format(MAX_TIME))



def make_structures():
    '''
    Need to run get_data() first to know MAX_TIME.

    Sets the frameTimes (np array) of length NUM_FRAMES, and the molPositions
    dictionary containing an empty list for each time in frameTimes
    '''
    global frameTimes, molPositions, MAX_TIME, NUM_FRAMES

    frameTimes = np.linspace(start=0, stop=MAX_TIME, num=NUM_FRAMES, endpoint=True)

    molPositions = {}
    for t in frameTimes:
        molPositions.update( {t : []} ) #each list is to be filled with (r, z) tuples

# =============================================================================
# Script for running this on cluster rather than Spyder
# =============================================================================
if __name__ == '__main__':

    if CLUSTER==True:
        directory=''

        initialize()
        main()
        writeData(directory+outfile)
