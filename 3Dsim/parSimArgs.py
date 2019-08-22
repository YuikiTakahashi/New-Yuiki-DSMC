#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:43:30 2018

@author: Dave

This is an auxiliary file containing only a specific set of parameters and
functionalities desired for processing on the HPC cluster.
"""

import numpy as np
import scipy.stats as st
import scipy.interpolate as si
from joblib import Parallel, delayed
from multiprocessing import Pool
import itertools
import argparse
import sys
# import matplotlib.pyplot as plt
from collections import defaultdict

#Set as 1 for basic vel_pdf
#Set as 2 for vel_corrected_pdf
SIMPLE_FLAG = 1 #DEFAULT is 1 i.e. simplest approximation



#Label all known geometries and map to a tuple (default_aperture, default_endPos)
#  1. default_aperture: gives the z position (mm) of what we take to be the aperture \
#                       in this geometry, used to tell when to start recording the
#                       particle location when LITE_MODE is true
#  2. default_endPos: gives the z position (mm) of the "end" point of the simulation
#                       site, i.e. where to stop the computation if the molecule gets there.
knownGeometries = {\
                   'fCell' : (0.064, 0.12),\
                   'gCell' : (0.064, 0.12),\
                   'hCell' : (0.064, 0.24),\
                   'jCell' : (0.064, 0.24),\
                   'kCell' : (0.064, 0.24),\
                   'mCell' : (0.081, 0.24),\
                   'nCell' : (0.073, 0.14),\
                   'pCell' : (0.0726, 0.2),\
                   'qCell' : (0.064, 0.12),\
                   'rCell' : (0.064, 0.12)\
                   }

collProb = 0.1 #Probability of collision, set at 1/10


class CallBack(object):
    completed = defaultdict(int)

    def __init__(self, index, parallel):
        self.index = index
        self.parallel = parallel

    def __call__(self, index):
        CallBack.completed[self.parallel] += 1
        print("Done with {}".format(CallBack.completed[self.parallel]))
        if self.parallel._original_iterable:
            self.parallel.dispatch_next()
import joblib.parallel
joblib.parallel.CallBack = CallBack

def set_derived_quants():
    '''
    This function must be run whenever n, T, T_s, vx, vy, or vz are changed.
    '''
    global U, vMean, vMeanM, coll_freq, dt, no_collide
    vMean = 2 * (2 * kb * T / (m * np.pi))**0.5
    vMeanM = 2 * (2 * kb * T_s / (M * np.pi))**0.5
    coll_freq = n * cross * vMean # vRel
    if collProb / coll_freq < 1e-4:
        dt = collProb / coll_freq # ∆t satisfying E[# collisions in 10∆t] = 1.
        no_collide = False
    else: # Density is so low that collision frequency is near 0
        no_collide = True # Just don't collide.


def get_flow_chars(filename):
    '''
    Retrieves the cell geometry and flowrate (in SCCM) from the FF filename.
    The cell geometry is crucial for the simulation to work!
    '''
    global flowrate, geometry

    #e.g. filename = flows/G_Cell/DS2g020
    if filename[13:16] == "DS2":
        geometry = {'f':"fCell", 'g':"gCell", 'h':"hCell", 'j':"jCell",\
                    'k':"kCell", 'm':"mCell", 'n':"nCell", 'p':"pCell",\
                    'q':"qCell", 'r':"rCell"}[filename[16]]
        flowrate = int(filename[17:20])

    else:
        raise ValueError('Could not recognize the DS2 flow file')
    print(geometry)
    print(flowrate)

# =============================================================================
# Probability Distribution Functions
# =============================================================================

# Maxwell-Boltzmann Velocity Distribution for ambient molecules
# Used temporarily since high-precision coll_vel_pdf distributions are too slow
class vel_pdf(st.rv_continuous):
    def _pdf(self,x):
        return (m/(2*np.pi*kb*T))**1.5 * 4*np.pi * x**2 * np.exp(-m*x**2/(2*kb*T))

class vel_corrected_pdf(st.rv_continuous):
    def _pdf(self,x):
        return (m**2)/(2 * kb**2 * T**2) * x**3 * np.exp(-m*x**2/(2*kb*T)) #extra factor of v
# Maxwell-Boltzmann Velocity Distribution for species molecules
# Used exclusively for setting initial velocities at a specified T_s
class species_vel_pdf(st.rv_continuous):
    def _pdf(self,x):
        return (M/(2*np.pi*kb*T_s))**1.5 * 4*np.pi * x**2 * np.exp(-M*x**2/(2*kb*T_s))


def coll_vel_pdf(x, y, z):
    '''
    For a given vector of *velocity* <x, y, z>, return the probability density of
    an ambient atom having that velocity given it was involved in a collision.
    '''
    sqspeed = x**2 + y**2 + z**2
    rel = ((x-(vx-xFlow))**2 + (y-(vy-yFlow))**2 + (z-(vz-zFlow))**2)**0.5
    vel = (m/(2*np.pi*kb*T))**1.5 * 4*np.pi * np.exp(-m*sqspeed/(2*kb*T))
    return rel * vel

# Define a PDF ~ sin(x) to be used for random determination of azimuthal velocity angle
class theta_pdf(st.rv_continuous):
    def _pdf(self,x):
        return np.sin(x)/2  # Normalized over its range [0, pi]


# Define a PDF ~ cos(x) to be used for random determination of impact angle
class Theta_pdf(st.rv_continuous):
    def _pdf(self,x):
        return -np.cos(x)  # Normalized over its range [pi/2, pi]

#==============================================================================
#Form-dependent parameter setup, must have axis-of-symmetry "wall" data
#==============================================================================


def dsmcQuant(x0, y0, z0, func):
    quant0 = func(z0, (x0**2 + y0**2)**0.5)[0][0]
    if func == f3:
        Vz = quant0
        vr = dsmcQuant(x0, y0, z0, f4)
        vPerpCw = dsmcQuant(x0, y0, z0, f5)

        theta = np.arctan2(y0, x0)
        rot = np.pi/2 - theta
        Vx = np.cos(rot) * vPerpCw + np.sin(rot) * vr
        Vy = -np.sin(rot) * vPerpCw + np.cos(rot) * vr

        return Vx, Vy, Vz
    if func == f1:
        return np.exp(quant0)
    return quant0

def inBounds(x, y, z, form='box', endPos=0.12):
    '''
    Return Boolean value for whether or not a position is within
    the boundary of "form".
    '''
    r = np.sqrt(x**2+y**2)

    if form in ['box', 'curvedFlowBox']:
        inside = abs(x) <= 0.005 and abs(y) <= 0.005 and abs(z) <= 0.005

    elif form == 'fCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.0635
        in2 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in3 = r < 0.030 and z >= 0.0640 and z < endPos
        inside = in1 + in2 + in3
        return inside

    elif form == 'gCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.05965
        in2 = r < 0.066-z and z > 0.05965 and z < 0.0635
        in3 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in4 = r < 0.030 and z >= 0.0640 and z <  endPos
        inside = in1 + in2 + in3 + in4
        return inside

    elif form == 'hCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.05965
        in2 = r < 0.066-z and z > 0.05965 and z < 0.0635
        in3 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in4 = r < z-0.0615 and z > 0.0640 and z < 0.06785
        in5 = r < 0.030 and z >= 0.06785 and z < endPos #Remember to extend endPos!
        inside = in1 + in2 + in3 + in4 + in5

    elif form == 'jCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.05965
        in2 = r < 0.066-z and z > 0.05965 and z < 0.0635
        in3 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in4 = r < (3.85/7.9)*(z-0.064)+0.0025 and z > 0.0640 and z < 0.0719
        in5 = r < 0.030 and z >= 0.0719 and z < endPos #Remember to extend endPos!
        inside = in1 + in2 + in3 + in4 + in5
        return inside

    elif form == 'kCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.0624
        in2 = r < (38.5/11)*(0.0624-z)+0.00635 and z > 0.0624 and z < 0.0635
        in3 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in4 = r < (3.85/7.9)*(z-0.064)+0.0025 and z > 0.0640 and z < 0.0719
        in5 = r < 0.030 and z >= 0.0719 and z < endPos #Remember to extend endPos!
        inside = in1 + in2 + in3 + in4 + in5
        return inside

    elif form == 'mCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.0635
        in2 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in3 = r < 0.009 and z > 0.064 and z < 0.068
        in4 = r < 0.00635 and z > 0.068 and z < 0.07275
        in5 = r < 0.009 and z > 0.07275 and z < 0.07575
        in6 = r < 0.00635 and z > 0.07575 and z < 0.0805
        in7 = r < 0.0025 and z > 0.0805 and z < 0.081
        in8 = r < 0.030 and z >= 0.081 and z < endPos
        inside = in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8
        return inside

    elif form == 'nCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.0635
        in2 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in3 = r < 0.009 and z > 0.064 and z < 0.066
        in4 = r < 0.00635 and z > 0.066 and z < 0.068
        in5 = r < 0.009 and z > 0.068 and z < 0.070
        in6 = r < 0.00635 and z > 0.070 and z < 0.072
        in7 = r < 0.0025 and z > 0.072 and z < 0.073
        in8 = r < 0.030 and z >= 0.073 and z < endPos
        inside = in1 + in2 + in3 + in4 + in5 + in6 + in7 + in8
        return inside

    elif form == 'pCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.0635
        in2 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in3 = r < 0.009 and z > 0.0640 and z < 0.067
        in4 = r < 0.00635 and z > 0.067 and z < 0.0721
        in5 = r < 0.0025 and z > 0.0721 and z < 0.0726
        in6 = r < 0.030 and z >= 0.0726 and z < endPos
        inside = in1 + in2 + in3 + in4 + in5 + in6
        return inside

    elif form == 'qCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.0635
        in2a = r < 0.0025 and z > 0.0635 and z < 0.064
        in2b = r < 0.00635 and r > 0.00585 and z > 0.0635 and z < 0.064
        in3 = r < 0.030 and z >= 0.064 and z < endPos
        inside = in1 + in2a + in2b + in3
        return inside

    elif form == 'rCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.0635
        in2 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in3 = r < 0.030 and z >= 0.0640 and z < endPos
        inside = in1 + in2 + in3
        return inside

    else:
        raise ValueError('Could not find bounds for geometry {}'.format(form))

def setAmbientFlow(x, y, z, form='box'):
    '''
    Given position of species molecule, set ambient flow velocities to known
    (DSMC-generated) local values.
    '''
    global xFlow, yFlow, zFlow
    if form in ['box', 'open']:
        xFlow, yFlow, zFlow = 0, 0, 0
    elif form == 'curvedFlowBox':
        r = (x**2+y**2)**0.5
        radFlow = -5*z*np.exp(-0.4*abs(5*z+1)) * 100 * r
        xFlow = x * radFlow / r * 100
        yFlow = y * radFlow / r * 100
        zFlow = 0.2 * 100
    elif form in knownGeometries:
        xFlow, yFlow, zFlow = dsmcQuant(x, y, z, f3)
        if abs(xFlow) > 1000:
            print(x, y, z, xFlow, 'm/s')

def setAmbientDensity(x, y, z, form='box'):
    '''
    Given position of species molecule, set ambient density to known
    (DSMC-generated) local value.
    '''
    global n
    if form in ['box', 'curvedFlowBox', 'open']:
        n = n
    elif form in knownGeometries:
        n = dsmcQuant(x, y, z, f1)
        if abs(n) > 1e26:
            print(x, y, z, n, 'm-3')

def setAmbientTemp(x, y, z, form='box'):
    '''
    Given position of species molecule, set ambient temperature to known
    (DSMC-generated) local value.
    '''
    global T
    if form in ['box', 'curvedFlowBox', 'open']:
        T = T
    elif form in knownGeometries:
        T = dsmcQuant(x, y, z, f2)
        if abs(T) > 500:
            print(x, y, z, T, 'K')

def updateParams(x, y, z, form='box'):
    setAmbientFlow(x, y, z, form)
    setAmbientDensity(x, y, z, form)
    setAmbientTemp(x, y, z, form)
    set_derived_quants()

# =============================================================================
# Helper Functions
# =============================================================================

def coll_vel_index_pdf(x, y, z, p, w):
    '''
    Here, indices x, y, z are linearly converted to velocities using a
    scale factor "precision" and offset "width/2" and passed into coll_vel_pdf.
    Returns probability density at the velocity corresponding to these indices.
    '''
    return coll_vel_pdf(p*(x-w/2), p*(y-w/2), p*(z-w/2))

def getAmbientVelocity(precision=4, width=700, simple=1):
    '''
    Returns the total ambient particle velocity from species rest frame.
    Each thermal velocity component can range from -width/2 to width/2 (m/s).
    Allowed thermal velocity components are spaced by "precision" (m/s).

    SIMPLE 1: for testing, uses vel_pdf instead of coll_vel_pdf.

    SIMPLE 2: uses vel_corrected_pdf, which has an extra factor of v
    '''
    if simple == 1:
        v0 = vel_cv.rvs()
        theta = theta_cv.rvs()
        phi = np.random.uniform(0, 2*np.pi)
        Vx, Vy, Vz = (v0*np.sin(theta)*np.cos(phi), v0*np.sin(theta)\
                           *np.sin(phi), v0*np.cos(theta))
        return Vx + xFlow - vx, Vy + yFlow - vy, Vz + zFlow - vz

    #Slightly more refined case. Rather than sample from a regular Maxwell-Boltzmann,
    #we attach an extra factor of v to the PDF, to account for collision frequency depending
    #on relative velocity, but approximate the molecule speed as negligible.
    elif simple == 2:
        v0 = vel_corrected_cv.rvs()
        theta = theta_cv.rvs()
        phi = np.random.uniform(0, 2*np.pi)
        Vx, Vy, Vz = (v0*np.sin(theta)*np.cos(phi), v0*np.sin(theta)\
                           *np.sin(phi), v0*np.cos(theta))
        return Vx + xFlow - vx, Vy + yFlow - vy, Vz + zFlow - vz



    width = int(width/precision) # scale for mesh
    probs = np.fromfunction(coll_vel_index_pdf, (width, width, width), \
                            p=precision, w=width).flatten()
    inds = np.linspace(0, len(probs)-1, len(probs))
    choice = np.random.choice(inds, p=probs/np.sum(probs))
    ind3D = np.where(np.reshape(inds, (width, width, width)) == choice)
    z_ind, y_ind, x_ind = int(ind3D[0]), int(ind3D[1]), int(ind3D[2])
    z, y, x = precision*(z_ind - width/2), precision*(y_ind - width/2), precision*(x_ind - width/2)
    return x + xFlow - vx, y + yFlow - vy, z + zFlow - vz

def initial_species_velocity(mode=1):
    '''
    Given species temperature, return randomized (Boltzmann) speed in a
    randomized (spherically uniform) direction.
    '''
    global T_s

    #These initial conditions assume the molecules begin thermalized with the 4K environment
    if mode in [0, 1, 9, 11]:
        T_s0 = 4

    #This initial condition is meant to approximate the post-ablation species
    #velocity distribution
    elif mode in [2]:
        T_s0 = 5000

    T_s_holder = T_s
    T_s = T_s0
    set_derived_quants() #Required because species_vel_cv draws temperature directly from T_s
    v0 = species_vel_cv.rvs()
    T_s = T_s_holder # Return to thermalized value
    set_derived_quants()

    if mode in [0,1,9,11]:

        theta = theta_cv.rvs()
        phi = np.random.uniform(0, 2*np.pi)
        Vx, Vy, Vz = (v0*np.sin(theta)*np.cos(phi), v0*np.sin(theta)\
                           *np.sin(phi), v0*np.cos(theta))
    elif mode in [2]:

        Vx, Vy, Vz = v0, 0, 0

    return Vx, Vy, Vz

def initial_species_position(L=0.01, form='', mode=1):
    '''
    Return a random position in a cube of side length L around the origin.
    '''
    global knownGeometries

    if form not in knownGeometries:
        x = np.random.uniform(-L/2, L/2)
        y = np.random.uniform(-L/2, L/2)
        z = np.random.uniform(-L/2, L/2)

    else:
        #Larger initial distribution of particles
        if mode==1:
            r = np.random.uniform(0, 0.004)
            ang = np.random.uniform(0, 2*np.pi)
            x, y = r * np.cos(ang), r * np.sin(ang)
            z = np.random.uniform(0.030, 0.040)

        #Standard initial distribution
        elif mode==0:
            r = np.random.uniform(0, 0.002)
            ang = np.random.uniform(0, 2*np.pi)
            x, y = r * np.cos(ang), r * np.sin(ang)
            z = np.random.uniform(0.035, 0.045)

        #Full-cell initial distribution for PROBE MODE in cell F
        elif mode==9:
            r = np.random.uniform(0,0.00635)
            ang = np.random.uniform(0, 2*np.pi)
            x, y = r * np.cos(ang), r * np.sin(ang)
            z = np.random.uniform(0.015,0.0635)

        #Full-cell initial distribution for PROBE MODE in cell H
        elif mode==11:
            z = np.random.uniform(0.015,0.0635)
            ang = np.random.uniform(0, 2*np.pi)
            if z <= 0.05965:
                r = np.random.uniform(0,0.00635)
            else:
                r = np.random.uniform(0,0.066-z)
            x, y = r * np.cos(ang), r * np.sin(ang)

        #Approximating ablation: 5mm width in the z direction, starting from the wall
        elif mode==2:
            x, y = -0.00635+0.0001, 0
            z = np.random.uniform(0.035,0.040)

        else:
            raise ValueError('Did not recognize INIT_MODE {}'.format(mode))

    return x, y, z

def collide():
    '''
    For current values of position (giving ambient flow rate) and velocity,
    increment vx, vy, vz according to collision physics.
    '''
    global vx, vy, vz
    Theta = Theta_cv.rvs()
    Phi = np.random.uniform(0, 2*np.pi)
    vx_amb, vy_amb, vz_amb = getAmbientVelocity(precision=3, simple=SIMPLE_FLAG)
    v_amb = (vx_amb**2 + vy_amb**2 + vz_amb**2)**0.5
    B = (vy_amb**2 + vz_amb**2 + (vx_amb-v_amb**2/vx_amb)**2)**-0.5

    vx += (v_amb * massParam * np.cos(Theta) * \
           (np.sin(Theta) * np.cos(Phi) * B * (vx_amb-v_amb**2/vx_amb)\
            + vx_amb * np.cos(Theta)/v_amb))

    vy += (v_amb * massParam * np.cos(Theta) * \
           (np.sin(Theta)*np.cos(Phi)*B*vy_amb + np.sin(Theta)*np.sin(Phi)*\
            (vz_amb/v_amb*B*(vx_amb-v_amb**2/vx_amb)-vx_amb*B*vz_amb/v_amb)\
            + np.cos(Theta)*vy_amb/v_amb))

    vz += (v_amb * massParam * np.cos(Theta) * \
           (np.sin(Theta)*np.cos(Phi)*B*vz_amb + np.sin(Theta)*np.sin(Phi)*\
            (vx_amb*B*vy_amb/v_amb-vy_amb/v_amb*B*(vx_amb-v_amb**2/vx_amb))\
            + np.cos(Theta)*vz_amb/v_amb))


# =============================================================================
# Simulation & Data Retrieval
# =============================================================================

#Returns traj, a list of strings containing position, velocity and times
#of the trajectory
#Gab: To every traj.append, added either a sim_time or a zero
def endPosition(extPos=0.12):
    '''
    Return the final position of a particle somewhere in the cell or else
    past the aperture within a distance extPos.
    '''

    #LITE_MODE: if true, only write data to file once at the beginning, and when past the aperture.
    #PROBE_MODE: if true, only write two lines per particle: initial and final
    #PARTICLE_NUMBER: total number of particles to simulate.


    #Important: need to properly retrieve geometry
    global vx, vy, vz, LITE_MODE, geometry, INIT_MODE, PROBE_MODE, default_aperture

    print("Running, INIT = {}".format(INIT_MODE))

    traj = []
    np.random.seed()
    x, y, z = initial_species_position(L=.01, form=geometry, mode=INIT_MODE)
    vx, vy, vz = initial_species_velocity(mode=INIT_MODE) #Thermalized to 4K environment
    sim_time = 0.0 #Tracking simulation time

    traj.append(' '.join(map(str, [round(1000*x,3), round(1000*y,3), round(1000*z,2), \
                                        round(vx,2), round(vy,2), round(vz,2), round(1000*sim_time,4) ] ) )+'\n')

    #Iterate updateParams() and update the particle position
    while inBounds(x, y, z, geometry, extPos):
        # Typically takes few ms to leave box
        updateParams(x, y, z, geometry)
        if np.random.uniform() < collProb and no_collide==False: # 1/10 chance of collision

            collide()

            #Print the full trajectory ONLY if 1) LITE_MODE=False, so we want all data,
            #or if 2) we are close enough to the aperture that we want to track regardless
            if (LITE_MODE == False or z > default_aperture - 0.0005) and PROBE_MODE == False:
                    traj.append(' '.join(map(str, [round(1000*x,3), round(1000*y,3), round(1000*z,2), \
                                                round(vx,2), round(vy,2), round(vz,2), round(1000*sim_time, 4) ] ) )+'\n')

        x += vx * dt
        y += vy * dt
        z += vz * dt
        sim_time += dt

    if z > extPos:
        # Linearly backtrack to boundary
        sim_time -= (z-extPos) / vz

        z = extPos
        x -= (z-extPos)/(vz * dt) * (vx * dt)
        y -= (z-extPos)/(vz * dt) * (vy * dt)


    traj.append(' '.join(map(str, [round(1000*x,3), round(1000*y,3), round(1000*z,2), \
                                   round(vx,2), round(vy,2), round(vz,2), round(1000*sim_time,4) ] ) )+'\n')
    traj.append(' '.join(map(str, [0,0,0,0,0,0,0]))+'\n') #Added an extra zero

    return traj


# =============================================================================
# Iterating endPosition
# =============================================================================

def showWalls():
    '''
    Generate a scatter plot of final positions of molecules as determined by
    the endPosition function parameters.
    '''
    print("Started showWalls")
    print("Running flowfield %s"%FF)
    print("Simulating {0} particles, cross-section multiplier {1}".format(PARTICLE_NUMBER, crossMult))

    f = open(outfile, "w+")

    global geometry, INIT_MODE, PROBE_MODE, default_aperture

    if geometry in knownGeometries:
        default_aperture = knownGeometries[geometry][0]
        default_endPos = knownGeometries[geometry][1]

    else:
        print("Failed: Did not recognize geometry")
        sys.exit()

    #N=(PARTICLE_NUMBER) different jobs, each with the parameter endPos set to default_endPos
    inputs = np.ones(PARTICLE_NUMBER) * default_endPos

    results = Parallel(n_jobs=-1,max_nbytes=None,verbose=50)(delayed(endPosition)(i) for i in inputs)
#    with Pool(processes=100) as pool:
#        results = pool.map(endPosition, inputs, 1)
    f.write('x (mm)   y (mm)   z (mm)   vx (m/s)   vy (m/s)   vz (m/s)   time (ms)   dens\n')
    f.write(''.join(map(str, list(itertools.chain.from_iterable(results)))))
    f.close()

def plot_boundaries(endPoint=0.12):
    global geometry

    size=500

    z_axis = np.linspace(0, endPoint, num=size)
    r_axis = np.linspace(0, 0.03, num=size)

    zv, rv = np.meshgrid(z_axis, r_axis)
    inOrOut = np.ones(zv.shape)

    for i in range(size):
        for j in range(size):
            inOrOut[i,j] = inBounds(x=0, y=rv[i,j], z=zv[i,j], form=geometry, endPos=endPoint)

    plt.pcolormesh(zv, rv, inOrOut)
    plt.show()
    sys.exit()
# =============================================================================
# Had to wrap the script into a main method, otherwise parallelization ran
# into issues when running the program on Windows
# =============================================================================


if __name__ == '__main__':

    print("Started main")

    parser = argparse.ArgumentParser('Specify simulation params')
    parser.add_argument('-ff', '--one', metavar='flows/X_Cell/DS2xyyy.DAT', help='File containing flowfield from DS2V output') # Specify flowfield
    parser.add_argument('-out', '--two', metavar='xyyy.dat', help = 'Output file name, to store trajectory data') # Specify output filename

    parser.add_argument('--mult', type=float, dest='mult', action='store', help='Multiplier for the collision cross section') # Specify cross section multiplier (optional)
    parser.add_argument('--npar', type=int, dest='npar', action='store', help='Number of particles to simulate') #Specify number of particles to simulate (optional, defaults to 1)
    parser.add_argument('--lite', dest='lite', action='store_true', help = 'Set TRUE if recording trajectories inside the cell is not necessary')

    parser.add_argument('--init_mode', type=int, dest='init_mode', action='store', help='Code number for initial particle distributions')
    parser.add_argument('--probe_mode', dest='probe_mode', action='store_true', help='Set TRUE if only particles final locations are needed')
    parser.set_defaults(lite=False, mult=5, npar=1, init_mode=0, probe_mode=False) #Defaults to LITE_MODE=False, 1 particle and crossMult=5
    args = parser.parse_args()

    FF = args.one
    outfile = args.two

    PARTICLE_NUMBER = args.npar
    crossMult = args.mult
    LITE_MODE = args.lite
    INIT_MODE = args.init_mode
    PROBE_MODE = args.probe_mode

    print("Particle number {0}, crossmult {1}, LITE_MODE {2}, INIT_MODE {3}".format(PARTICLE_NUMBER,crossMult,LITE_MODE, INIT_MODE))
    print("PROBE_MODE {}".format(PROBE_MODE))
    print("SIMPLE_FLAG = {}".format(SIMPLE_FLAG))
    # =============================================================================
    # Constant Initialization
    # =============================================================================

    kb = 1.38 * 10**-23
    NA = 6.022 * 10**23
    T = 4 # Ambient temperature (K)
    T_s = 4 # Species temperature (K) for initial velocity distributions
    m = 0.004 / NA # Ambient gas mass (kg)
    M = .190061 / NA # Species mass (kg)
    massParam = 2 * m / (m + M)
    n = 10**21 # m^-3
    cross = 4 * np.pi * (140 * 10**(-12))**2 # two-helium cross sectional area
    cross *= 4 # Rough estimate of He-YbOH cross sectional area
    cross *= crossMult # Manual adjustment to vary collision frequency

    # Global variable initialization: these values are irrelevant
    vx, vy, vz, xFlow, yFlow, zFlow = 0, 0, 0, 0, 0, 0

    set_derived_quants()

    vel_cv = vel_pdf(a=0, b=4*vMean, name='vel_pdf') # vel_cv.rvs()
    vel_corrected_cv = vel_corrected_pdf(a=0,b=4*vMean, name='vel_corrected_pdf') #Implementing approximate Bayesian approach i.e. extra factor of v

    species_vel_cv = species_vel_pdf(a=0, b=5*vMeanM, name='species_vel_pdf') # species_vel_cv.rvs()
    theta_cv = theta_pdf(a=0, b=np.pi, name='theta_pdf') # theta_cv.rvs() for value
    Theta_cv = Theta_pdf(a=np.pi/2, b=np.pi, name='Theta_pdf') # Theta_cv.rvs() for value

    # =============================================================================
    # Form-dependent parameter setup
    # =============================================================================

    # =============================================================================
    # Must have axis-of-symmetry "wall" data in FF for this to work
    # =============================================================================
    try:
        flowField = np.loadtxt(FF, skiprows=1) # Assumes only first row isn't data.
        get_flow_chars(FF)
        # plot_boundaries(endPoint=0.2)
        global geometry, flowrate, default_aperture

        #geometry = 'fCell'
        #flowrate = int(FF[-7:-4])
        print("Loading flow field: geometry {0}, flowrate = {1} SCCM".format(geometry,flowrate))

        zs, rs, dens, temps = flowField[:, 0], flowField[:, 1], flowField[:, 2], flowField[:, 7]
        #print("1")
        vzs, vrs, vps = flowField[:, 4], flowField[:, 5], flowField[:, 6]
        quantHolder = [zs, rs, dens, temps, vzs, vrs, vps]
        #print("2")

        if geometry in ['fCell', 'gCell', 'nCell', 'qCell', 'rCell']:
            grid_x, grid_y = np.mgrid[0.010:0.12:4500j, 0:0.030:1500j] # high density, to be safe.
        elif geometry in ['hCell', 'jCell', 'kCell', 'mCell']:
            grid_x, grid_y = np.mgrid[0.010:0.24:9400j, 0:0.030:1500j] # high density, to be safe.
        elif geometry in ['pCell']:
            grid_x, grid_y = np.mgrid[0.010:0.20:9400j, 0:0.030:1500j] # high density, to be safe.
        else:
            print('No geometry')
            sys.exit()

        print("Block 1: density and temperature")
        grid_dens = si.griddata(np.transpose([zs, rs]), np.log(dens), (grid_x, grid_y), 'nearest')
        grid_temps = si.griddata(np.transpose([zs, rs]), temps, (grid_x, grid_y), 'nearest')

        print("Block 2: velocities")

        grid_vzs = si.griddata(np.transpose([zs, rs]), vzs, (grid_x, grid_y), 'nearest')
        grid_vrs = si.griddata(np.transpose([zs, rs]), vrs, (grid_x, grid_y), 'nearest')
        grid_vps = si.griddata(np.transpose([zs, rs]), vps, (grid_x, grid_y), 'nearest')

        print("Interpolating")
        # These are interpolation functions:
        f1 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_dens)
        f2 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_temps)
        f3 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vzs)
        f4 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vrs)
        f5 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vps)

        print("Done loading flow field")
    except:
        print("Note: No Flow Field DSMC data.")
        sys.exit()

    #print("Step 3")
    showWalls()
    print("Done: flowfield {0}, multiplier {1}".format(FF,crossMult))
