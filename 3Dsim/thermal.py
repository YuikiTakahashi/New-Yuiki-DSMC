import numpy as np
import matplotlib.pyplot as plt

#directory = 'C:/Users/gabri/Desktop/HutzlerSims/Gas-Simulation/3Dsim/Data/'
#file_ext = 'traj017d.dat'

directory="C:/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/ThermalHeavy/F_Cell/"
file_ext = 'f100_th.dat'

#Data for x, y, z comes in mm
z0 = 0.015*1000
endPos = 0.120*1000
LZ = endPos - z0
RMAX = 0.030*1000

#Sets grid resolution in the Z and R axes
NZ = 1000
NR = 1000

#This is for running on the cluster
CLUSTER = True
WRITE_FILE = "THERMALDATA.dat"

def main():
    global f, numParticles, VR_LISTS, VZ_LISTS, V_LISTS


    ptally = 0

    ltally = 0

    vr_temp, vz_temp, v_temp = [], [], []

    prev_loc = None
    current_loc = (0,0)

    #Iterate through every line in the particle data file
    for i in range(len(f)):


        #If reach a row of zeros, reset for the next particle and save previous buffer array
        if not np.any(f[i]):
            ptally += 1 #Keeping track of particles analyzed so far
            ltally += 1
            if round(100*ptally/numParticles,1)%5 == 0:
                print("Analyzed particle {0} of {1}: {2}% done".format(ptally, numParticles,round(100*ptally/numParticles,2)))

            VR_LISTS.update( {prev_loc:VR_LISTS[prev_loc]+[np.mean(vr_temp)] } )
            VZ_LISTS.update( {prev_loc:VZ_LISTS[prev_loc] + [np.mean(vz_temp)]} )
            V_LISTS.update( {prev_loc:V_LISTS[prev_loc] + [np.mean(v_temp)]} )
            prev_loc = None #Reset location
            vr_temp, vz_temp, v_temp = [], [], [] #Clear buffer arrays

        #If we are beginning logging for a new particle, need to guess grid location
        else:
            x, y, z, vx, vy, vz, tim = f[i]

            if np.sqrt(x**2+y**2) > RMAX:
                #Do nothing. Ignore this line
                ltally += 1

            elif prev_loc == None:

                ltally += 1

                r = np.sqrt(x**2+y**2)
                vr = np.sqrt(vx**2 + vy**2)
                try:
                    current_loc = zRegion(z=z, prev_k=210), rRegion(r=r, prev_k=10) #Guess initial grid location hard-coded
                except ValueError as err:
                    print(err.args)
                    print("Stopped on line {}".format(ltally))
                print("Line {}, ".format(ltally), end='')
                print("Initial loc was {}".format(current_loc))

                prev_loc = current_loc
                vr_temp.append(vr)
                vz_temp.append(vz)
                v_temp.append(np.sqrt(vr**2 + vz**2))


            #Continue logging for current particle, can tell where to look
            else:
                ltally +=1
                print("Line {}, ".format(ltally), end='')

                r = np.sqrt(x**2+y**2)
                vr = np.sqrt(vx**2 + vy**2)
                try:
                    current_loc = zRegion(z, prev_loc[0]), rRegion(r, prev_loc[1])
                except ValueError as err:
                    print(err.args)
                    print("Stopped on line {}".format(ltally))
                print("loc {}".format(current_loc))

                #If we are in the same sector of the grid, just add data to the buffer arrays
                if current_loc == prev_loc:
                    vr_temp.append(vr)
                    vz_temp.append(vz)
                    v_temp.append(np.sqrt(vr**2 + vz**2))

                elif current_loc != prev_loc:
                    #Take means of the buffer arrays and store them in the previous location of the map
                    VR_LISTS.update( {prev_loc:VR_LISTS[prev_loc]+[np.mean(vr_temp)] } )
                    VZ_LISTS.update( {prev_loc:VZ_LISTS[prev_loc] + [np.mean(vz_temp)]} )
                    V_LISTS.update( {prev_loc:V_LISTS[prev_loc] + [np.mean(v_temp)]} )

                    #Now clear the buffer arrays and start logging info for the new location
                    vr_temp = [vr]
                    vz_temp = [vz]
                    v_temp = [np.sqrt(vr**2 + vz**2)]

                    prev_loc = current_loc

# =============================================================================
# Taking V*_LISTS and averaging them into 2D numpy arrays
# =============================================================================

def compileData():
    '''
    Compiles data into NZxNR numpy arrays VRDATA, VZDATA and VDATA.
    Useful for loading data into arrays in Spyder.
    '''
    global VRDATA, VZDATA, VDATA, NZ, NR, V_LISTS, VR_LISTS, VZ_LISTS
    VRDATA, VZDATA, VDATA = np.zeros((NZ,NR)), np.zeros((NZ,NR)), np.zeros((NZ,NR))

    for i in range(NZ):
        for j in range(NR):

            if len(VR_LISTS[(i,j)]) > 0:
                VRDATA[i][j] = np.mean(VR_LISTS[(i,j)])
                VZDATA[i][j] = np.mean(VZ_LISTS[(i,j)])
                VDATA[i][j] = np.mean(V_LISTS[(i,j)])


def writeData(writeFile):
    '''
    Writes data matrix from V*_LISTS to a file.
    Does not require compileData() to be run beforehand.
    '''
    global NZ, NR, V_LISTS, VR_LISTS, VZ_LISTS

    with open(writeFile, 'a+') as tw:
        tw.write('Z   R   VR   VZ   V\n')

        for i in range(NZ):
            for j in range(NR):

                if len(VR_LISTS[(i,j)]) == 0:
                    tw.write('   '.join(map(str, [i, j, 0, 0, 0] ))+'\n')
                else:
                    vr = np.mean(VR_LISTS[(i,j)])
                    vz = np.mean(VZ_LISTS[(i,j)])
                    v = np.mean(V_LISTS[(i,j)])

                    tw.write('   '.join(map(str, [i, j, round(vr,3), round(vz,3), round(v,3)] ))+'\n')
    tw.close()




def showPlots(which):
    global VRDATA, VZDATA, VDATA
    arr = {'vr':VRDATA, 'vz':VZDATA, 'v':VDATA}[which]
    img = plt.imshow(arr)
    plt.show()

# =============================================================================
# Initializing data matrix, velocity maps and z/r axes
# =============================================================================
def initialize():
    global directory, file_ext

    print('Getting data from {}...'.format(file_ext), end='')
    get_data(directory+file_ext)

    print('Making velocity maps, {}x{} grid'.format(NR,NZ))
    make_vlists()

    print('Done initializing')



def get_data(file):
    global f, numParticles

    f = np.loadtxt(file, skiprows=1)

    numParticles = 0
    for i in range(len(f)):
        if not np.any(f[i]):
            numParticles += 1 # count number of particles simulated in file

    print(' {} particles'.format(numParticles))



def make_vlists():
    global V_LISTS, VR_LISTS, VZ_LISTS, z_axis, r_axis, z0, LZ, RMAX, NZ, NR

    VR_LISTS, VZ_LISTS, V_LISTS = {}, {}, {}

    z_axis = np.ones(NZ+1)
    r_axis = np.ones(NR+1)

    for i in range(NZ):
        for j in range(NR):
            V_LISTS.update(  {(i,j) : []} )
            VR_LISTS.update( {(i,j) : []} )
            VZ_LISTS.update( {(i,j) : []} )

    for i in range(NZ+1):
        z_axis[i] = z0 + i*(LZ / NZ)

    for i in range(NR+1):
        r_axis[i] = i*(RMAX / NR)



# =============================================================================
# Methods for locating particles on the grid
# =============================================================================
def zRegion(z, prev_k):
    global z_axis, NZ

    delta = 0
    while delta < NZ+1:

        #Check on the right
        if prev_k+delta+1 < len(z_axis) and z_axis[prev_k + delta] < z and z <= z_axis[prev_k + 1 + delta]:
            return prev_k + delta

        #Check on the left
        elif prev_k-delta >= 0 and z_axis[prev_k - delta] < z and z <= z_axis[prev_k + 1 - delta]:
            return prev_k - delta

        #Search next neighbor
        else:
            delta += 1

    #If we exit while loop
    raise ValueError('Overflowed delta', 'z: {}, prev_zk: {}'.format(z,prev_k))

def rRegion(r, prev_k):
    global r_axis, NR

    delta = 0
    while delta < NR+1:

        #Check above
        if prev_k+delta+1 < len(r_axis) and r_axis[prev_k + delta] < r and r <= r_axis[prev_k + 1 + delta]:
            return prev_k + delta

        #Check below
        elif prev_k-delta >= 0 and r_axis[prev_k - delta] < r and r <= r_axis[prev_k + 1 - delta]:
            return prev_k - delta

        #Search next neighbor
        else:
            delta += 1

    #If we exit while loop
    raise ValueError('Overflowed delta', 'r: {}, prev_rk: {}'.format(r,prev_k))


if __name__ == '__main__':

    if CLUSTER==True:
        directory=''

        initialize()
        main()
        writeData(WRITE_FILE)
