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

parser = argparse.ArgumentParser('Simulation Specs')
parser.add_argument('-ff', '--one') # Specify flowfield
parser.add_argument('-out', '--two') # Specify output filename
parser.add_argument('--mult', type=int) # Specify cross section multiplier (optional)
args = parser.parse_args()

FF = args.one
outfile = args.two
if args.mult:
    crossMult = args.mult
else:
    crossMult = 5


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

def set_derived_quants():
    '''
    This function must be run whenever n, T, T_s, vx, vy, or vz are changed.
    '''
    global U, vMean, vMeanM, coll_freq, dt, no_collide
    vMean = 2 * (2 * kb * T / (m * np.pi))**0.5
    vMeanM = 2 * (2 * kb * T_s / (M * np.pi))**0.5
    coll_freq = n * cross * vMean # vRel
    if 0.1 / coll_freq < 1e-4:
        dt = 0.1 / coll_freq # ∆t satisfying E[# collisions in 100∆t] = 1.
        no_collide = False
    else: # Density is so low that collision frequency is near 0
        no_collide = True # Just don't collide.

set_derived_quants()


# =============================================================================
# Probability Distribution Functions
# =============================================================================

# Maxwell-Boltzmann Velocity Distribution for ambient molecules
# Used temporarily since high-precision coll_vel_pdf distributions are too slow
class vel_pdf(st.rv_continuous):
    def _pdf(self,x):
        return (m/(2*np.pi*kb*T))**1.5 * 4*np.pi * x**2 * np.exp(-m*x**2/(2*kb*T))
vel_cv = vel_pdf(a=0, b=4*vMean, name='vel_pdf') # vel_cv.rvs()

# Maxwell-Boltzmann Velocity Distribution for species molecules
# Used exclusively for setting initial velocities at a specified T_s
class species_vel_pdf(st.rv_continuous):
    def _pdf(self,x):
        return (M/(2*np.pi*kb*T_s))**1.5 * 4*np.pi * x**2 * np.exp(-M*x**2/(2*kb*T_s))
species_vel_cv = species_vel_pdf(a=0, b=5*vMeanM, name='species_vel_pdf') # species_vel_cv.rvs()

def coll_vel_pdf(x, y, z):
    '''
    For a given vector of *velocity* <x, y, z>, return the probability density of
    an ambient molecule having that velocity given it was involved in a collision.
    '''
    sqspeed = x**2 + y**2 + z**2
    rel = ((x-(vx-xFlow))**2 + (y-(vy-yFlow))**2 + (z-(vz-zFlow))**2)**0.5
    vel = (m/(2*np.pi*kb*T))**1.5 * 4*np.pi * np.exp(-m*sqspeed/(2*kb*T))
    return rel * vel

# Define a PDF ~ sin(x) to be used for random determination of azimuthal velocity angle
class theta_pdf(st.rv_continuous):
    def _pdf(self,x):
        return np.sin(x)/2  # Normalized over its range [0, pi]
theta_cv = theta_pdf(a=0, b=np.pi, name='theta_pdf') # theta_cv.rvs() for value

# Define a PDF ~ cos(x) to be used for random determination of impact angle
class Theta_pdf(st.rv_continuous):
    def _pdf(self,x):
        return -np.cos(x)  # Normalized over its range [pi/2, pi]
Theta_cv = Theta_pdf(a=np.pi/2, b=np.pi, name='Theta_pdf') # Theta_cv.rvs() for value

# =============================================================================
# Form-dependent parameter setup
# =============================================================================

# =============================================================================
# Must have axis-of-symmetry "wall" data in FF for this to work !!!
# =============================================================================
try:
    flowField = np.loadtxt(FF, skiprows=1) # Assumes only first row isn't data.
    zs, rs, dens, temps = flowField[:, 0], flowField[:, 1], flowField[:, 2], flowField[:, 7]
    vzs, vrs, vps = flowField[:, 4], flowField[:, 5], flowField[:, 6]
    quantHolder = [zs, rs, dens, temps, vzs, vrs, vps]
    grid_x, grid_y = np.mgrid[0.010:0.12:4500j, 0:0.030:1500j] # high density, to be safe.
    grid_dens = si.griddata(np.transpose([zs, rs]), np.log(dens), (grid_x, grid_y), 'nearest')
    grid_temps = si.griddata(np.transpose([zs, rs]), temps, (grid_x, grid_y), 'nearest')
    grid_vzs = si.griddata(np.transpose([zs, rs]), vzs, (grid_x, grid_y), 'nearest')
    grid_vrs = si.griddata(np.transpose([zs, rs]), vrs, (grid_x, grid_y), 'nearest')
    grid_vps = si.griddata(np.transpose([zs, rs]), vps, (grid_x, grid_y), 'nearest')
    # These are interpolation functions:
    f1 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_dens)
    f2 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_temps)
    f3 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vzs)
    f4 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vrs)
    f5 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vps)
except:
    print("Note: No Flow Field DSMC data.")

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
    if form in ['box', 'curvedFlowBox']:
        inside = abs(x) <= 0.005 and abs(y) <= 0.005 and abs(z) <= 0.005
    elif form == 'currentCell':
        r = np.sqrt(x**2+y**2)
        in1 = r < 0.00635 and z > 0.015 and z < 0.0635
        in2 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in3 = r < 0.030 and z >= 0.0640 and z < endPos
        inside = in1 + in2 + in3
    return inside

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
    elif form == 'currentCell':
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
    elif form == 'currentCell':
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
    elif form == 'currentCell':
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

def getAmbientVelocity(precision=4, width=700, simple=True):
    '''
    Returns the total ambient particle velocity from species rest frame.
    Each thermal velocity component can range from -width/2 to width/2 (m/s).
    Allowed thermal velocity components are spaced by "precision" (m/s).
    SIMPLE case: for testing, uses vel_pdf instead of coll_vel_pdf.
    '''
    if simple == True:
        v0 = vel_cv.rvs()
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

def initial_species_velocity(T_s0):
    '''
    Given species temperature, return randomized (Boltzmann) speed in a
    randomized (spherically uniform) direction.
    '''
    global T_s
    T_s_holder = T_s
    T_s = T_s0
    set_derived_quants()
    v0 = species_vel_cv.rvs()
    T_s = T_s_holder # Return to thermalized value
    set_derived_quants()
    theta = theta_cv.rvs()
    phi = np.random.uniform(0, 2*np.pi)
    Vx, Vy, Vz = (v0*np.sin(theta)*np.cos(phi), v0*np.sin(theta)\
                       *np.sin(phi), v0*np.cos(theta))
    return Vx, Vy, Vz

def initial_species_position(L=0.01, form=''):
    '''
    Return a random position in a cube of side length L around the origin.
    '''
    if form != 'currentCell':
        x = np.random.uniform(-L/2, L/2)
        y = np.random.uniform(-L/2, L/2)
        z = np.random.uniform(-L/2, L/2)
    else:
        r = np.random.uniform(0, 0.002)
        ang = np.random.uniform(0, 2*np.pi)
        x, y = r * np.cos(ang), r * np.sin(ang)
        z = np.random.uniform(0.035, 0.045)
    return x, y, z

def collide():
    '''
    For current values of position (giving ambient flow rate) and velocity,
    increment vx, vy, vz according to collision physics.
    '''
    global vx, vy, vz
    Theta = Theta_cv.rvs()
    Phi = np.random.uniform(0, 2*np.pi)
    vx_amb, vy_amb, vz_amb = getAmbientVelocity(precision=3, simple=True)
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

def endPosition(extPos=0.12):
    '''
    Return the final position of a particle somewhere in the cell or else
    past the aperture within a distance extPos.
    '''
    global vx, vy, vz
    traj = []
    np.random.seed()
    x, y, z = initial_species_position(.01, 'currentCell')
    vx, vy, vz = initial_species_velocity(T_s0=4)
    traj.append(' '.join(map(str, [round(1000*x,3), round(1000*y,3), round(1000*z,2), \
                                        round(vx,2), round(vy,2), round(vz,2)]))+'\n')
    while inBounds(x, y, z, 'currentCell', extPos):
        # Typically takes few ms to leave box
        updateParams(x, y, z, 'currentCell')
        if np.random.uniform() < 0.1 and no_collide==False: # 1/10 chance of collision
            collide()
            traj.append(' '.join(map(str, [round(1000*x,3), round(1000*y,3), round(1000*z,2), \
                                        round(vx,2), round(vy,2), round(vz,2)]))+'\n')
        x += vx * dt
        y += vy * dt
        z += vz * dt

    if z > extPos:
        # Linearly backtrack to boundary
        z = extPos
        x -= (z-extPos)/(vz * dt) * (vx * dt)
        y -= (z-extPos)/(vz * dt) * (vy * dt)
    traj.append(' '.join(map(str, [round(1000*x,3), round(1000*y,3), round(1000*z,2), \
                                   round(vx,2), round(vy,2), round(vz,2)]))+'\n')
    traj.append(' '.join(map(str, [0,0,0,0,0,0]))+'\n')
    return traj


# =============================================================================
# Iterating endPosition
# =============================================================================

def showWalls():
    '''
    Generate a scatter plot of final positions of molecules as determined by
    the endPosition function parameters.
    '''
    f = open(outfile, "w")
    inputs = np.ones(100)*0.12
    results = Parallel(n_jobs=-1)(delayed(endPosition)(i) for i in inputs)
#    with Pool(processes=100) as pool:
#        results = pool.map(endPosition, inputs, 1)
    f.write('x (mm)   y (mm)   z (mm)   vx (m/s)   vy (m/s)   vz (m/s)\n')
    f.write(''.join(map(str, list(itertools.chain.from_iterable(results)))))
    f.close()

showWalls()