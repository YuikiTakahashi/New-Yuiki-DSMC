
import numpy as np
import scipy.stats as st
import scipy.interpolate as si
from joblib import Parallel, delayed
from multiprocessing import Pool
import itertools
import argparse
import sys

global z0, endPos, LZ, RMAX, NZ, NR, z_axis, r_axis

z0 = 0.015
endPos = 0.120
LZ = endPos - z0
RMAX = 0.00635

NZ = 1000
NR = 1000

z_axis = np.ones(NZ+1)
r_axis = np.ones(NR+1)


#This way, z_axis[k] = z_k and
#          r_axis[k] = r_k

for i in range(NZ+1):
    z_axis[i] = z0 + i*(LZ / NZ)

for i in range(NR+1):
    r_axis[i] = i*(RMAX / NR)

directory = '/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/'

f = np.loadtxt(directory+filename), skiprows=1)

numParticles = 0 #number of simulated particles
for i in range(len(f)):
    if not np.any(f[i]):
        numParticles += 1 # count number of particles simulated in file

VR_LISTS, VZ_LISTS, V_LISTS = {}, {}, {}, {}

for i in range(NZ):
    for j in range(NR):
        V_LISTS.update( {(i,j) : []} )
        VR_LISTS.update( {(i,j) : []} )
        VZ_LISTS.update( {(i,j) : []} )


vr_temp, vz_temp, v_temp = [], [], [], []

prev_loc = None
current_loc = (0,0)

for i in range(len(f)):

#NEED TO HANDLE CHANGE OF PARTICLES I.E. AFTER WE REACH A ROW OF ZEROS

    if np.any(f[i]):
        x, y, z, vx, vy, vz, tim = f[i-1]
        r = np.sqrt(x**2+y**2)
        vr = np.sqrt(vx**2 + vy**2)

        #Stores index (zk, rk) of z-region and r-region, for current particle location
        current_loc = zRegion(z, prev_loc[0]), rRegion(r, prev_loc[1])

        #If we are in the same sector of the grid, just add data to the buffer arrays
        if current_loc == prev_loc:
            vr_temp.append(vr)
            vz_temp.append(vz)
            v_temp.append(np.sqrt(vr**2 + vz**2))

        elif current_loc != prev_loc: #and the buffer is not empty
            #Take means of the buffer arrays and store them in the previous location of the map
            vrMean = np.mean(vr_temp)
            vzMean = np.mean(vz_temp)
            vMean = np.mean(v_temp)

            VR_LISTS[prev_loc].append(vrMean)
            VZ_LISTS[prev_loc].append(vzMean)
            V_LISTS[prev_loc].append(vMean)

            #Now clear the buffer arrays and start logging info for the new location
            vr_temp = [vr]
            vz_temp = [vz]
            v_temp = [np.sqrt(vr**2 + vz**2)]

            prev_loc = current_loc


def zRegion(z, prev_k):
    global z_axis

    delta = 0
    while True:

        #Check on the right
        if z_axis[prev_k + delta] < z and z < z_axis[prev_k + 1 + delta]:
            return prev_k + delta

        #Check on the left
        elif z_axis[prev_k - delta] < z and z < z_axis[prev_k + 1 - delta]:
            return prev_k - delta

        #Search next neighbor
        else:
            delta += 1

def rRegion(r, prev_r):
    global r_axis

    delta = 0
    while True:

        #Check above
        if r_axis[prev_k + delta] < r and r < r_axis[prev_k + 1 + delta]:
            return prev_k + delta

        #Check below
        elif r_axis[prev_k - delta] < r and r < r_axis[prev_k + 1 - delta]:
            return prev_k - delta

        #Search next neighbor
        else:
            delta += 1
