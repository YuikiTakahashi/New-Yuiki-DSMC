#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Modified by Yuiki Takahashi in 2020

Created August 2019
@author: Gabriel

Contains the main simulation class, ParticleTracing() to carry out parallelized
molecule path-tracing simulations through a buffer gas (DSMC-generated) flow field.

Optimization left to do:
*Parallelization library (joblib is finnicky)
*Collision updating (can trim random number generation)
"""
#from keras import backend as K
#K.common.set_image_dim_ordering('th')

import math
import numpy as np
import scipy.stats as st
import scipy.interpolate as si
from joblib import Parallel, delayed
#from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool
import itertools
import argparse
import sys
import joblib.parallel
from scipy.stats import uniform
from numpy import random, linalg

###############################################################################
# CONSTANTS
###############################################################################

KB = 1.38 * 10**-23 #Boltzmann
NA = 6.022 * 10**23 #Avogadro
M_HE = 0.004 / NA # Helium gas particle mass (kg)
M_S = .190061 / NA # Species mass (kg)
M_RED = (M_HE * M_S) / (M_HE + M_S)
MASS_PARAM = 2 * M_HE / (M_HE + M_S)
#CROSS_YBF = 20 * 4 * np.pi * (140 * 10**(-12))**2 # helium-helium cross section
CROSS_YBF = 10*100*10**(-20) #YbF-He cross section at 20 K (m^2) 
# YbOH mass in amu
mass = 173 + 16 + 1  
# buffer gass mass (He) in amu
bgmass = 4

# self._m = 0.004 / NA # Helium gas particle mass (kg)
# self._M = .190061 / NA # Species mass (kg)
# self._massParam = 2 * m / (m + M)

###############################################################################
# SIMULATION CLASS
###############################################################################
class ParticleTracing_Yuiki(object):
    '''
    Main simulation class. Simulates path of a YbOH molecule through
    a helium buffer gas cell.

    Stores all physical parameters and quantities, random variables, buffer
    gas flow field, etc.
    '''

    def __init__(self, flowFieldName='flows/F_Cell/DS2f005.DAT', NPAR=10, CROSS_MULT=1, LITE_MODE=True, INIT_COND=0, PROBE_MODE=False, CORRECTION=1, HEATMAP=False):

        #Set as 0 for basic Maxwell-Boltzmann PDF
        #Set as 1 for corrected PDF
        self._CORRECTION = CORRECTION
        
        self._HEATMAP = HEATMAP

        #Label all known geometries and map to a tuple (default_aperture, default_endPos)
        #  1. default_aperture: gives the z position (mm) of what we take to be the aperture \
        #                       in this geometry, used to tell when to start recording the
        #                       particle location when LITE_MODE is true
        #  2. default_endPos: gives the z position (mm) of the "end" point of the simulation
        #                       site, i.e. where to stop the computation if the molecule gets there.
        self._knownGeometries = {\
                           'fCell' : (0.064, 0.12),\
                           'gCell' : (0.064, 0.12),\
                           'hCell' : (0.064, 0.24),\
                           'jCell' : (0.064, 0.24),\
                           'kCell' : (0.064, 0.24),\
                           'mCell' : (0.081, 0.24),\
                           'nCell' : (0.073, 0.14),\
                           'pCell' : (0.0726, 0.2),\
                           'qCell' : (0.064, 0.12),\
                           'rCell' : (0.064, 0.12),\
                           'yCell' : (0.053, 0.12),\
                           'tCell' : (0.0886, 0.18)\
                           }

        #Probability of collision, set at 1/10. Higher probability means finer time-step
        self._collProb = 0.1

        # =============================================================================
        # Load cell geometry and flow field
        # =============================================================================
        self._FF = flowFieldName #Filename for DSMC output file
        self.set_flow_type(flowFieldName)


        self._PARTICLE_NUMBER = NPAR #Number of particles to simulate
        self._CROSS_MULT = CROSS_MULT #Fine-tuning colllision cross section parameter

        self._LITE_MODE = LITE_MODE #If True, does not record particle trajectory until close to the aperture.

        self._INIT_COND = INIT_COND #Initial conditions "type" selected for the molecules
        self._PROBE_MODE = PROBE_MODE #If True, records the molecules initial and final positions ONLY

        # =============================================================================
        # Use the CROSS_MULT parameter to tune the He-Species collision cross section.
        # =============================================================================

        self._cross = CROSS_YBF # Rough estimate of He-YbOH cross section area.

        # vMean = 2 * (2 * KB * 4 / (M_HE * np.pi))**0.5 #Mean velocity for 4K helium gas
        # self._theta_cv = theta_pdf(a=0, b=np.pi, name='theta_pdf') # theta_cv.rvs() for value
        self._Theta_cv = Theta_pdf(a=np.pi/2, b=np.pi, name='Theta_pdf') # Theta_cv.rvs() for value
        # ***** End of class constructor ****** #

    def set_flow_type(self, FF):
        '''
        Loads DSMC output file FF and uses bivariate spline method to set gas characteristics
        on a grid of points.

        Args:
        FF is the filename of the DSMC output file (DS2FF.DAT) which should be named in the format
        xyyy.dat, where x denotes the geometry and yyy denotes the gas flow-rate. e.g. f005.dat

        The class instance will update the following attributes:
        # self._geometry: cell geometry, denoted by a single letter ('f', 'g', 'h', etc)
        # self._flowrate: buffer gas flowrate of the DSMC simulation, in SCCM.

        # self._fDens, self._fTemp, self._fVz, self._fVr, self._fVp: the scipy.interpolate objects
        corresponding to the density, temperature, etc. on the grid.

        **IMPORTANT: this method needs to be updated with each new designed cell geometry, depending
        on the desired size and coarseness of the simulation site.
        '''
        try:
            #Geometry is a string in ['fCell', 'gCell', 'hCell', ..., 'pCell', 'rCell', ...]
            #Flowrate is an integer in [2, 5, ..., 200] (SCCM)

            geometry, flowrate = get_flow_chars(self._FF)

            self._geometry = geometry
            self._flowrate = flowrate
            # print("Loading flow field: geometry {0}, flowrate {1} SCCM".format(geometry,self._flowrate))

            flowField = np.loadtxt(self._FF, skiprows=1) # Assumes only first row is not data.

            zs, rs, dens, temps = flowField[:, 0], flowField[:, 1], flowField[:, 2], flowField[:, 7] #Arrays of z and r coordinates, density and temperature
            vzs, vrs, vps = flowField[:, 4], flowField[:, 5], flowField[:, 6] #velocity field in z, r, and "perpendicular" directions
            quantHolder = [zs, rs, dens, temps, vzs, vrs, vps]

            if geometry in ['fCell', 'gCell', 'nCell', 'qCell', 'rCell']:
                grid_x, grid_y = np.mgrid[0.010:0.12:2250j, 0:0.030:750j] # high density, to be safe. Was 4500 x 1500 points
            elif geometry in ['hCell', 'jCell', 'kCell', 'mCell']:
                grid_x, grid_y = np.mgrid[0.010:0.24:9400j, 0:0.030:1500j] # high density, to be safe.
            elif geometry in ['pCell']:
                grid_x, grid_y = np.mgrid[0.010:0.20:9400j, 0:0.030:1500j] # high density, to be safe.
            elif geometry in ['tCell', 'sCell', 'yCell']: # cases added by Ben, 2/19/2020
                grid_x, grid_y = np.mgrid[0.005:0.12:4500j, 0:0.030:1000j]
            else:
                raise ValueError('Unknown geometry')

            print("Loading grids ... ")
            grid_dens = si.griddata(np.transpose([zs, rs]), np.log(dens + 1.0), (grid_x, grid_y), 'nearest') # the +1.0 inside the log is to make sure the log-density never goes negative.
            grid_temps = si.griddata(np.transpose([zs, rs]), temps, (grid_x, grid_y), 'nearest')
            grid_vzs = si.griddata(np.transpose([zs, rs]), vzs, (grid_x, grid_y), 'nearest')
            grid_vrs = si.griddata(np.transpose([zs, rs]), vrs, (grid_x, grid_y), 'nearest')
            grid_vps = si.griddata(np.transpose([zs, rs]), vps, (grid_x, grid_y), 'nearest')

            print("Interpolating ... ",end='')
            self._fDens = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_dens, kx=1, ky=1)
            self._fTemp = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_temps, kx=1, ky=1)
            self._fVz = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vzs, kx=1, ky=1)
            self._fVr = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vrs, kx=1, ky=1)
            self._fVp = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vps, kx=1, ky=1)

            print("Flow field loaded")

        except:
            raise ValueError('Could not load flow field')

    def set_sim_params(self, NPAR=None, CROSS_MULT=None, LITE_MODE=None, INIT_COND=None, PROBE_MODE=None, CORRECTION=None, HEATMAP=None):
        '''
        Update/reset the ParticleTracing object parameters
        '''

        if HEATMAP != None:
            self._HEATMAP = HEATMAP #Number of particles to simulate
            
        if NPAR != None:
            self._PARTICLE_NUMBER = NPAR #Number of particles to simulate

        if CROSS_MULT != None:
            self._CROSS_MULT = CROSS_MULT #Fine-tuning colllision cross section parameter
            self._cross = CROSS_YBF # Rough estimate of He-YbOH cross section area.

        if LITE_MODE != None:
            self._LITE_MODE = LITE_MODE #If True, does not record particle trajectory until close to the aperture.

        if INIT_COND != None:
            self._INIT_COND = INIT_COND #Initial conditions "type" selected for the molecules

        if PROBE_MODE != None:
            self._PROBE_MODE = PROBE_MODE #If True, records the molecules initial and final positions ONLY

        if CORRECTION != None:
            self._CORRECTION = CORRECTION

###############################################################################
# MAIN PROGRAM METHODS
###############################################################################

    def get_trajectory(self, boundary):
        '''
        To be run in parallel in the self.parallel_main class method.
        Computes one molecule trajectory, returns it as list.
        '''
        return self.track_molecules(endPos=boundary, numTraj=1)


    def parallel_main(self, outfile):
        '''
        Uses parallelization library to compute several molecule
        trajectories and write the results to an output file.
        '''

        #Boundary of the simulation site, i.e. when do we terminate the particle's path.
        default_endPos = self._knownGeometries[self._geometry][1]
        
        HEATMAP = self._HEATMAP

        inputs = np.ones(self._PARTICLE_NUMBER) * default_endPos
        f = open(outfile, "w+")

        results = Parallel(n_jobs=-1,max_nbytes=None,verbose=50)(delayed(self.get_trajectory)(i) for i in inputs)
        
        if HEATMAP != True:
            f.write('x (mm)   y (mm)   z (mm)   vx (m/s)   vy (m/s)   vz (m/s)   time (ms)   dens\n')
        elif HEATMAP == True:
            f.write('r (mm)   z (mm)   extraction (0=N/1=Y)   time (ms)\n')
        
        f.write(''.join(map(str, list(itertools.chain.from_iterable(results)))))
        f.close()

        # pool = Pool()
        # tempResults = pool.amap(self.get_trajectory, inputs).get()
        # return tempResults


    # def showWalls(self, outfile):
    #     '''
    #     Generate a scatter plot of final positions of molecules as determined by
    #     the endPosition function parameters.
    #     '''
    #     print("Started showWalls")
    #
    #     #The knownGeometries array stores the default end position for each geometry
    #     #as one of the parameters.
    #     default_endPos = self._knownGeometries[self._geometry][1]
    #
    #     #N=(PARTICLE_NUMBER) jobs, each with the parameter endPos set to default_endPos
    #     inputs = np.ones(self._PARTICLE_NUMBER) * default_endPos
    #
    #     results = Parallel(n_jobs=-1,max_nbytes=None,verbose=50)(delayed(self.get_trajectory)(i) for i in inputs)
    #     #    with Pool(processes=100) as pool:
    #     #        results = pool.map(endPosition, inputs, 1)
    #
    #     f = open(outfile, "w+")
    #     f.write('x (mm)   y (mm)   z (mm)   vx (m/s)   vy (m/s)   vz (m/s)   time (ms)   dens\n')
    #     f.write(''.join(map(str, list(itertools.chain.from_iterable(results)))))
    #     f.close()


    def nonparallel_main(self, outfile):
#        print("CORRECTION = {}".format(self._CORRECTION))

        particleNum = self._PARTICLE_NUMBER
        cellGeometry = self._geometry
        default_endPos = self._knownGeometries[cellGeometry][1]
        f = open(outfile, "w+")

        tempResults = self.track_molecules(endPos=default_endPos, numTraj=particleNum)
        f.write('x (mm)   y (mm)   z (mm)   vx (m/s)   vy (m/s)   vz (m/s)   time (ms)   dens\n')
        f.write(''.join(map(str, list(itertools.chain.from_iterable(tempResults)))))
        f.close()
#        return tempResults


    #Returns traj, a list of strings containing position, velocity and times
    def track_molecules(self, endPos, numTraj):
        '''
        Simulate path of a molecule through the buffer gas cell.

        Arguments:
        endPos -- location on the z-axis (in meters) where the simulation stops
        numTraj -- number of particle trajectories to simulate

        Returns:
        traj -- list of strings. Each string is a row containing particle
                trajectory data, organized by [x, y, z, vx, vy, vz, t]

        Path is terminated once the molecule hits a cell wall, or else reaches
        the simulation boundary set at z = endPos (m).
        '''

        # If true, only write data to file once at the beginning, and when close
        # to aperture, i.e. don't record particle trajectory for most of the inner
        # cell. This is to make the output files lighter.
        LITE_MODE = self._LITE_MODE

        # If true, only write two lines per particle: initial and final
        PROBE_MODE = self._PROBE_MODE
        
        HEATMAP = self._HEATMAP

        cellGeometry = self._geometry
        default_aperture = self._knownGeometries[cellGeometry][0] #z location of cell aperture
        traj = []

        np.random.seed()

        #Compute [numTraj] trajectories
        if HEATMAP != True:

            for i in range(numTraj):

                #Initialize molecule and start simulation "clock"
                x, y, z = self.initial_species_position()
                vx, vy, vz = self.initial_species_velocity()
                simTime = 0.0

                bgFlow, bgDens, bgTemp = self.update_buffer_gas(x, y, z)
                v_thermal = 2 * (2 * KB * bgTemp / (M_HE * np.pi))**0.5
                v_flow = ((bgFlow[0]-vx)**2+(bgFlow[1]-vy)**2+(bgFlow[2]-vz)**2)
                v_mean = v_thermal + v_flow*4/(3*np.pi*v_thermal)
                dt = self._collProb / (bgDens * self._cross * v_mean)

                #Start recording particle trajectory
                traj.append(' '.join(map(str, [round(1000*x,3), round(1000*y,3), round(1000*z,2), \
                                                    round(vx,2), round(vy,2), round(vz,2), round(1000*simTime,4) ] ) )+'\n')

                #Iterate updateParams() and update the particle position
                while inBounds(x, y, z, cellGeometry, endPos):

                    bgFlow, bgDens, bgTemp = self.update_buffer_gas(x, y, z)
                    dt, no_collide = self.get_derived_quants(temp=bgTemp, dens=bgDens, prev_dt=dt, \
                                                         v_rel=np.linalg.norm((bgFlow-np.array([vx, vy, vz]))))
                    
                    if uniform.rvs(0,1) < self._collProb and no_collide==False: # 1/10 chance of collision

                        vx, vy, vz = self.collide(temp=bgTemp, v_flow=bgFlow, v_mol=np.array([vx, vy, vz]))

                        #Print the full trajectory ONLY if 1) LITE_MODE=False, so we want all data,
                        #or if 2) we are close enough to the aperture that we want to track regardless
                        if (LITE_MODE == False or z > default_aperture - 0.0005) and PROBE_MODE == False:
                                traj.append(' '.join(map(str, [round(1000*x,3), round(1000*y,3), round(1000*z,2), \
                                                            round(vx,2), round(vy,2), round(vz,2), round(1000*simTime, 4) ] ) )+'\n')

                    x += vx * dt
                    y += vy * dt
                    z += vz * dt
                    simTime += dt

                if z > endPos:
                    # Linearly backtrack to boundary
                    simTime -= (z-endPos) / vz
                    z = endPos
                    x -= (z-endPos)/(vz * dt) * (vx * dt)
                    y -= (z-endPos)/(vz * dt) * (vy * dt)


                traj.append(' '.join(map(str, [round(1000*x,3), round(1000*y,3), round(1000*z,2), \
                                               round(vx,2), round(vy,2), round(vz,2), round(1000*simTime,4) ] ) )+'\n')
                traj.append(' '.join(map(str, [0,0,0,0,0,0,0]))+'\n') #Added an extra zero

                #Trajectory is over, loop over this [numTraj] times
                
                
        elif HEATMAP == True:
            for i in range(numTraj):

                #Initialize molecule and start simulation "clock"
                xini, yini, zini = self.initial_species_position()
                vx, vy, vz = self.initial_species_velocity()
                simTime = 0.0 
                x = xini
                y = yini
                z = zini
                rini = np.sqrt(xini**2 + yini**2)

                bgFlow, bgDens, bgTemp = self.update_buffer_gas(x, y, z)
                #v_mean = 2 * (2 * KB * bgTemp / (M_HE * np.pi))**0.5
                v_thermal = 2 * (2 * KB * bgTemp / (M_HE * np.pi))**0.5
                v_flow = ((bgFlow[0]-vx)**2+(bgFlow[1]-vy)**2+(bgFlow[2]-vz)**2)
                v_mean = v_thermal + v_flow*4/(3*np.pi*v_thermal)
                dt = self._collProb / (bgDens * self._cross * v_mean)


                #Iterate updateParams() and update the particle position
                while inBounds(x, y, z, cellGeometry, default_aperture):

                    bgFlow, bgDens, bgTemp = self.update_buffer_gas(x, y, z)
                    dt, no_collide = self.get_derived_quants(temp=bgTemp, dens=bgDens, prev_dt=dt, \
                                                         v_rel=np.linalg.norm((bgFlow-np.array([vx, vy, vz]))))

                    if uniform.rvs(0,1) < self._collProb and no_collide==False: # 1/10 chance of collision

                        vx, vy, vz = self.collide(temp=bgTemp, v_flow=bgFlow, v_mol=np.array([vx, vy, vz]))

                    x += vx * dt
                    y += vy * dt
                    z += vz * dt
                    simTime += dt

                if z > default_aperture:
                    # Linearly backtrack to boundary
                    simTime -= (z-default_aperture) / vz
                    #z = default_aperture
                    #x -= (z-default_aperture)/(vz * dt) * (vx * dt)
                    #y -= (z-default_aperture)/(vz * dt) * (vy * dt)
                    
                    #Start recording particle trajectory
                    traj.append(' '.join(map(str, [round(1000*rini,3), round(1000*zini,2), 1, round(1000*simTime,4) ] ) )+'\n')
                
                else:
                    #simTime = float("NaN")
                    traj.append(' '.join(map(str, [round(1000*rini,3), round(1000*zini,2), 0, round(1000*simTime,4) ] ) )+'\n')

        return traj

################################################################################
# Handling buffer gas properties and derived quantities
################################################################################

    def get_derived_quants(self, temp, dens, prev_dt, v_rel):
        '''
        Must be run any time n, T, T_s, or <vx,vy,vz> are changed.
        '''
        T = temp #Buffer gas temperature
        n = dens #Buffer gas number density
        m = M_HE #mass of He
        
        if T > 0.1:
            cross = self._cross #He-YbOH cross section
            collProb = self._collProb #Set probability of molecule-atom collision

            v_mean = 2 * (2 * KB * T / (m * np.pi))**0.5
            #v_flow = v_rel**2
            #v_mean = v_thermal + v_flow*4/(3*np.pi*v_thermal)
            coll_freq = n * cross * (v_mean + 4/(3*np.pi) * v_rel**2/v_mean)

            if collProb / coll_freq < 1e-4:
                dt = collProb / coll_freq # ∆t satisfying E[# collisions in 10∆t] = 1.
                no_collide = False

            # Density is so low that collision frequency is ~0, just don't collide
            else:
                dt = prev_dt
                no_collide = True
        else:
            dt = prev_dt
            no_collide = True 

        return dt, no_collide

    def update_buffer_gas(self, x, y, z):
        '''
        Given species molecule location, update buffer gas properties.

        Returns:
        v_flow -- gas bulk flow velocity (np.array[vx,vy,vz]) (m/s)
        n -- gas number density (float) (m^-3)
        T -- gas temperature (float) (Kelvins)
        '''
        v_flow = self.get_ambient_flow(x, y, z)
        n = self.get_ambient_density(x, y, z)
        T = self.get_ambient_temp(x, y, z)
        return v_flow, n, T

    def get_ambient_flow(self, x, y, z):
        '''
        Given position of species molecule, get ambient flow velocities from known
        (DSMC-generated) buffer gas velocity field.
        Returns:
        v_flow -- gas bulk flow velocity (np.array[vx,vy,vz]) (m/s)
        '''
        v_flow = self.dsmc_quant(x, y, z, 'flow')
        return v_flow

    def get_ambient_density(self, x, y, z):
        '''
        Given position of species molecule, get local density from known
        (DSMC-generated) buffer gas density field.
        Returns:
        n -- gas number density (float) (m^-3)
        '''
        n = self.dsmc_quant(x, y, z, 'dens')
        return n

    def get_ambient_temp(self, x, y, z):
        '''
        Given position of species molecule, get local temperature from known
        (DSMC-generated) buffer gas temperature field.
        '''
        temp = self.dsmc_quant(x, y, z, 'temp')
        return temp
    
    def dsmc_quant(self, x0, y0, z0, quant):
        '''
        Return quantity (quant) evaluated at the given location in the cell.

        Note: density is stored as log(dens) in self._fDens, so need to return exp().
        '''
        if quant == 'dens':
            logdens = self._fDens(z0, (x0**2 + y0**2)**0.5)[0][0]
            return np.exp(logdens)

        elif quant == 'temp':
            tem = self._fTemp(z0, (x0**2 + y0**2)**0.5)[0][0]
            return tem

        elif quant == 'flow':
            Vz = self._fVz(z0, (x0**2 + y0**2)**0.5)[0][0]
            vr = self._fVr(z0, (x0**2 + y0**2)**0.5)[0][0]
            vPerpCw = self._fVp(z0, (x0**2 + y0**2)**0.5)[0][0]
            theta = np.arctan2(y0, x0)
            rot = np.pi/2 - theta
            Vx = np.cos(rot) * vPerpCw + np.sin(rot) * vr
            Vy = -np.sin(rot) * vPerpCw + np.cos(rot) * vr
            return np.array([Vx, Vy, Vz])
        
    # Generate "num_points" random points in "dimension" that have uniform probability over the unit ball scaled by "radius" (length of points are in range [0, "radius"]).
    def random_ball(self, radius):

        # generate random directions by normalizing the length of a vector of random-normal values
        random_directions = random.normal(size=(3, 1))
        random_directions /= linalg.norm(random_directions, axis=0)

        # gaussian distrbiution over the radius with 3 sigma.
        random_radii = 100 
        while random_radii >3:
            random_radii = abs(random.normal(0,1,1)) 
        # Return the list of random (direction & length) points.
        return radius/3 * (random_directions * random_radii).T

    def random_ball2(self, radius):

        # generate random directions by normalizing the length of a vector of random-normal values
        random_directions = random.normal(size=(3, 1))
        random_directions /= linalg.norm(random_directions, axis=0)

        # uniform distribution over the radius.
        random_radii = uniform.rvs(0,1)
        
        # Return the list of random (direction & length) points.
        return radius * (random_directions * random_radii).T

    def initial_species_position(self):
        '''
        Return a random initial position for the molecule.
        Distribution of possible locations depends on which INIT_COND is active.
        * INIT_COND 0: (Small) Cylinder. 10mm x 4mm
        * INIT_COND 1: (Default) Cylinder. 10mm x 8mm
        * INIT_COND 2: 5000K ablation
        * INIT_COND 9: Full cell F
        * INIT_COND 11: Full cell H
        * INIT_COND 12: 1 cm diamter gaussian ball (the center is z = 0.064 m)
        * INIT_COND 13: 1 cm diamter uniform ball (the center is z = 0.064 m)
        '''
        #Retrieve which pre-specified type of initial conditions
        #were chosen for this simulation
        mode = self._INIT_COND
        HEATMAP = self._HEATMAP
        

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
        
        # 1 cm (= 0.01 m) diamter gasussian ball (the center of the ball is z = 0.064 m)
        elif mode==12:
            r = self.random_ball(0.005)
            x, y, znew = r[0]
            z = znew + 0.064
            
        # 1 cm (= 0.01 m) diamter uniform ball (the center of the ball is z = 0.064 m)    
        elif mode==13:
            r = self.random_ball2(0.005)
            x, y, znew = r[0]
            z = znew + 0.064
            
        # uniformly distributed over the t cell    
        elif mode==14:
            r = np.random.uniform(0,0.00635)
            ang = np.random.uniform(0, 2*np.pi)
            x, y = r * np.cos(ang), r * np.sin(ang)
            z = np.random.uniform(0.035,0.0881)
            
        elif HEATMAP == True and mode==15:
            r = 0.0005*np.random.randint(low=1,high=12)
            ang = np.random.uniform(0, 2*np.pi)
            x, y = r * np.cos(ang), r * np.sin(ang)
            z = 0.0005*np.random.randint(low=71,high=176)
            
            
        else:
            raise ValueError('Did not recognize INIT_COND {}'.format(mode))

        return x, y, z

    def initial_species_velocity(self):
        '''
        Given species temperature, return randomized (Boltzmann) speed in a
        randomized (spherically uniform) direction.

        * INIT_COND 0: 4K thermal velocity distribution, spherically uniform
        * INIT_COND 1: ditto
        * INIT_COND 9: ditto
        * INIT_COND 11: ditto

        * INIT_COND 2: 5000K thermal velocity distribution, straight out from cell wall
        '''
        # global T_s
        init_cond = self._INIT_COND
        #These initial conditions assume the molecules begin thermalized with the 4K environment
        if init_cond in [0, 1, 9, 11, 12, 13, 14, 15]:
            T_s = 4.0 #species temperature: assume 4K thermalization

            Vx = self.particle_generator(prop='Mol_Thermal_Vel', T=T_s)
            Vy = self.particle_generator(prop='Mol_Thermal_Vel', T=T_s)
            Vz = self.particle_generator(prop='Mol_Thermal_Vel', T=T_s)

            # v0 = self._max_boltz_cv.rvs(m=M_S, T=species_temp)
            # theta = self._theta_cv.rvs()
            # phi = np.random.uniform(0, 2*np.pi)
            # Vx, Vy, Vz = (v0*np.sin(theta)*np.cos(phi), v0*np.sin(theta)\
            #                    *np.sin(phi), v0*np.cos(theta))


        elif init_cond in [2]:
            '''
            Approximates post-ablation species
            velocity distribution
            '''
            T_s = 5000
            Vx = self.particle_generator(prop='Mol_Thermal_Vel', T=T_s)
            Vy, Vz = 0, 0

            # v0 = self._max_boltz_cv.rvs(m=M_S, T=species_temp)
            # Vx, Vy, Vz = v0, 0, 0

        return Vx, Vy, Vz

    def get_ambient_velocity(self, temp, v_flow, v_mol):
        '''
        Returns a random ambient particle velocity from species rest frame.

        Arguments:
        temp -- local buffer gas temperature (float)
        v_flow -- local buffer gas flow velocity (np.array[xFlow, yFlow, zFlow])
        v_mol -- current molecule velocity (np.array[vx,vy,vz])

        SIMPLE 1: uses MaxwellBoltzmann_pdf.
        SIMPLE 2: uses AmbientCorrected_pdf, which has an extra factor of v
        '''
        xFlow, yFlow, zFlow = v_flow[0], v_flow[1], v_flow[2]
        vx, vy, vz = v_mol[0], v_mol[1], v_mol[2]

        correc = self._CORRECTION

        if correc == 0:
            '''
            Simplest case: use unbiased Maxwell-Boltzmann thermal velocity distribution
            '''

            vxGas = xFlow + self.particle_generator(prop='He_Thermal_Vel', T=temp)
            vyGas = yFlow + self.particle_generator(prop='He_Thermal_Vel', T=temp)
            vzGas = zFlow + self.particle_generator(prop='He_Thermal_Vel', T=temp)
            return vxGas - vx, vyGas - vy, vzGas - vz

            # v0 = self._max_boltz_cv.rvs(m=M_HE, T=temp)
            # theta = self._theta_cv.rvs()
            # phi = np.random.uniform(0, 2*np.pi)
            # Vx, Vy, Vz = (v0*np.sin(theta)*np.cos(phi), v0*np.sin(theta)\
            #                    *np.sin(phi), v0*np.cos(theta))
            # return Vx + xFlow - vx, Vy + yFlow - vy, Vz + zFlow - vz

        elif correc == 2:
            '''DO NOT USE! Just for Debug'''
            v0 = self._max_boltz_cv.rvs(m=M_HE, T=temp)
            theta = self._theta_cv.rvs()
            phi = np.random.uniform(0, 2*np.pi)
            Vx, Vy, Vz = (v0*np.sin(theta)*np.cos(phi), v0*np.sin(theta)\
                               *np.sin(phi), v0*np.cos(theta))
            return Vx + xFlow - vx, Vy + yFlow - vy, Vz + zFlow - vz

        elif correc == 1:
            '''
            "Biased" thermal speed distribution. An extra factor of v enters the PDF,
            accounting for the velocity-dependent probability of collision.
            Assumes spherical symmetry as an approximation.
            '''

            # v0 = self._vel_ambient_cv.rvs(T=temp)
            # theta = self._theta_cv.rvs()
            # phi = np.random.uniform(0, 2*np.pi)
            # Vx, Vy, Vz = (v0*np.sin(theta)*np.cos(phi), v0*np.sin(theta)\
            #                    *np.sin(phi), v0*np.cos(theta))
            # return Vx + xFlow - vx, Vy + yFlow - vy, Vz + zFlow - vz

            vMag = self.particle_generator(prop='Coll_Rel_Vel',T=temp)
            theta = self.particle_generator(prop='Theta')
            phi = 2 * np.pi * np.random.uniform(0,1)

            vxGas = xFlow + vMag*np.sin(theta)*np.cos(phi)
            vyGas = yFlow + vMag*np.sin(theta)*np.sin(phi)
            vzGas = zFlow + vMag*np.cos(theta)
            return vxGas - vx, vyGas - vy, vzGas - vz
        

        else:
            raise ValueError('Unknown CORRECTION')

    def particle_generator(self, prop='He_Thermal_Vel', T=None):
        '''
        Generates a selected particle property from the appropriate distribution.

        Args:
        prop: property to be generated. Must be one of
                * 'He_Thermal_Vel': helium particle thermal speed (1 dimensional)
                * 'Mol_Thermal_Vel': molecular species thermal speed (1 dimensional)
                * 'Radial Position':
                * 'Coll_Rel_Vel': relative speed of colliding particles (with bias)

        T: temperature (required if generating a thermal velocity)

        Returns:
        a velocity v (float), if prop is one of 'He_Thermal_Vel', 'Mol_Thermal_Vel',
        or 'Coll_Rel_Vel'.
        '''

        if prop in ['He_Thermal_Vel', 'Mol_Thermal_Vel']:

            m = {'He_Thermal_Vel':M_HE, 'Mol_Thermal_Vel':M_S}[prop]
            k = KB
            coef = np.sqrt(2*k*T/m)

            r1, r2 = np.random.uniform(0,1,size=2)

            v = coef * np.sin(2*np.pi*r1) * np.sqrt(-1*np.log(r2))
            return v

        elif prop == 'Coll_Rel_Vel':
            '''
            Generate relative speed between colliding particles,
            in COM frame. Uses accept-reject technique.
            '''

            #Uses reduced mass \mu, of helium and molecular masses
            m = M_RED

            #Mean thermal velocity, used only to set the bounds of the PDF
            vMean = 2 * (2 * KB * T / (m * np.pi))**0.5

            #Maximum value of PDF
            fMax = 1.5 * np.sqrt(3*m/(KB*T)) * np.exp(-1.5)

            #Set the PDF as the collision velocity distribution with fixed temperature
            f = lambda x: collisionVelPDF(x, T=T)

            #Sample the PDF using accept-reject algorithm. The upper bound on the velocity
            #PDF is set at vMax=5*vMean, where the probability density should be negligible.
            v = accept_reject_gen(pdf=f, xmin=0, xmax=5*vMean, pmax=fMax)
            return v[0]

        elif prop == 'Theta':
            '''
            Return theta distributed like ~sin(theta)/2
            '''
            r1 = np.random.uniform(0,1)
            theta = np.arccos(1 - 2*r1)
            return theta

        else: raise ValueError('Incorrect property in generator')

#     def collide(self, temp, v_flow, v_mol):
#         '''
#         For current values of position (giving ambient flow rate) and velocity,
#         increment vx, vy, vz according to collision physics.
#         '''
#         massParam = MASS_PARAM

#         vx, vy, vz = v_mol[0], v_mol[1], v_mol[2]
#         Theta = self._Theta_cv.rvs()
#         Phi = np.random.uniform(0, 2*np.pi)

#         vx_amb, vy_amb, vz_amb = self.get_ambient_velocity(temp, v_flow, v_mol)

#         v_amb = (vx_amb**2 + vy_amb**2 + vz_amb**2)**0.5
#         B = (vy_amb**2 + vz_amb**2 + (vx_amb-v_amb**2/vx_amb)**2)**-0.5

#         vx += (v_amb * massParam * np.cos(Theta) * \
#                (np.sin(Theta) * np.cos(Phi) * B * (vx_amb-v_amb**2/vx_amb)\
#                 + vx_amb * np.cos(Theta)/v_amb))

#         vy += (v_amb * massParam * np.cos(Theta) * \
#                (np.sin(Theta)*np.cos(Phi)*B*vy_amb + np.sin(Theta)*np.sin(Phi)*\
#                 (vz_amb/v_amb*B*(vx_amb-v_amb**2/vx_amb)-vx_amb*B*vz_amb/v_amb)\
#                 + np.cos(Theta)*vy_amb/v_amb))

#         vz += (v_amb * massParam * np.cos(Theta) * \
#                (np.sin(Theta)*np.cos(Phi)*B*vz_amb + np.sin(Theta)*np.sin(Phi)*\
#                 (vx_amb*B*vy_amb/v_amb-vy_amb/v_amb*B*(vx_amb-v_amb**2/vx_amb))\
#                 + np.cos(Theta)*vz_amb/v_amb))

#         return vx, vy, vz
    
    
    
    def collide(self, temp, v_flow, v_mol):
        '''
        For current values of position (giving ambient flow rate) and velocity,
        increment vx, vy, vz according to collision physics. Assuming Hard-Sphere Scattering.
        '''

        vx, vy, vz = v_mol[0], v_mol[1], v_mol[2]

        vx_amb, vy_amb, vz_amb = self.get_ambient_velocity(temp, v_flow, v_mol)

        Cx = vx + vx_amb
        Cy = vy + vy_amb
        Cz = vz + vz_amb
        Wx = mass/(mass + bgmass)*vx + bgmass/(mass + bgmass)*Cx
        Wy = mass/(mass + bgmass)*vy + bgmass/(mass + bgmass)*Cy
        Wz = mass/(mass + bgmass)*vz + bgmass/(mass + bgmass)*Cz

        r1 = uniform.rvs(0,1)
        r2 = uniform.rvs(0,1)

        coschai = 2*r1-1
        sinchai = math.sqrt(1-coschai**2)
        theta = 2*math.pi*r2

        g = math.sqrt( (vx-Cx)**2 + (vy-Cy)**2 +(vz-Cz)**2 )
        gx = g*coschai
        gy = g*sinchai*math.cos(theta)
        gz = g*sinchai*math.sin(theta)

        v_colx = Wx + bgmass/(mass + bgmass)*gx - vx
        v_coly = Wy + bgmass/(mass + bgmass)*gy - vy
        v_colz = Wz + bgmass/(mass + bgmass)*gz - vz

        vx += v_colx

        vy += v_coly

        vz += v_colz

        return vx, vy, vz



def get_flow_chars(filename):
    '''
    Retrieves the cell geometry and flowrate (in SCCM) from the FF filename.
    This function relies on flow field naming scheme:
        Cell geometry X, flowrate abc:
        FILENAME = 'flows/X_Cell/DS2xabc.DAT'
    '''

    #e.g. filename = flows/G_Cell/DS2g020
    if filename[13:16] == "DS2" and (not filename[16] in ['t']):
        geometry = {'f':"fCell", 'g':"gCell", 'h':"hCell", 'j':"jCell",\
                    'k':"kCell", 'm':"mCell", 'n':"nCell", 'p':"pCell",\
                    'q':"qCell", 'r':"rCell", 'y':"yCell"}[filename[16]]
        flowrate = int(filename[17:20])
        
    elif filename[13:16] == "DS2" and (filename[16] in ['t']):
        geometry = 'tCell'
        flowrate = 1

    else:
        raise ValueError('Could not recognize the DS2 flow file')

    return geometry, flowrate

# =============================================================================
# Probability Distribution Functions
# =============================================================================

# Maxwell-Boltzmann Velocity Distribution for ambient molecules
# Used temporarily since high-precision coll_vel_pdf distributions are too slow
class MaxwellBoltzmann_pdf(st.rv_continuous):
    '''
    MaxwellBoltzmann_pdf SPEED distribution with mass m, temperature T
    '''
    def _pdf(self, x, m, T):
        return (m/(2*np.pi*KB*T))**1.5 * 4*np.pi * x**2 * np.exp(-m*x**2/(2*KB*T))

class AmbientCorrected_pdf(st.rv_continuous):
    '''
    Approximate distribution for ambient particle speeds.

    Maxwell-Boltzmann speed distribution times an extra factor of v,
    which takes into account the velocity dependence of the collisional
    cross section.
    '''
    def _pdf(self, x, T):
        m = M_HE #Should be changed to reduced mass?
        return (m**2)/(2 * KB**2 * T**2) * x**3 * np.exp(-m*x**2/(2*KB*T)) #extra factor of v

# Maxwell-Boltzmann Velocity Distribution for species molecules
# Used exclusively for setting initial velocities at a specified T_s
class species_vel_pdf(st.rv_continuous):
    def _pdf(self,x):
        return (M/(2*np.pi*kb*T_s))**1.5 * 4*np.pi * x**2 * np.exp(-M*x**2/(2*kb*T_s))


# Define a PDF ~ sin(x) to be used for random determination of azimuthal velocity angle
class theta_pdf(st.rv_continuous):
    '''
    For determining azimuthal velocity angle.
    '''
    def _pdf(self,x):
        return np.sin(x)/2  # Normalized over its range [0, pi]


# Define a PDF ~ cos(x) to be used for random determination of impact angle
class Theta_pdf(st.rv_continuous):
    def _pdf(self,x):
        return -np.cos(x)  # Normalized over its range [pi/2, pi]

def collisionVelPDF(x, T, m= M_RED):
    return (m**2)/(2 * KB**2 * T**2) * x**3 * np.exp(-m*x**2/(2*KB*T)) #PDF for relative speed, evaluated at y


def accept_reject_gen(pdf, n=1, xmin=0, xmax=1, pmax=None):
    pmin=0

    if pmax==None:
        x=np.linspace(xmin,xmax,1000)
        y=pdf(x)
        pmax=y.max()

    # Counters
    naccept=0
    ntrial=0

    # Keeps generating numbers until we achieve the desired n
    ran=[] # output list of random numbers
    while naccept<n:
        x=uniform.rvs(xmin,xmax) # x'
        y=uniform.rvs(pmin,pmax) # y'

        if y<pdf(x):
            ran.append(x)
            naccept += 1
        ntrial += 1

    ran=np.asarray(ran)

    return ran


def inBounds(x, y, z, cell=None, endPos=0.12):
    '''
    Return Boolean value for whether or not a position is within
    the boundary of "cell".
    '''
    r = np.sqrt(x**2+y**2)

    if cell == 'fCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.0635
        in2 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in3 = r < 0.030 and z >= 0.0640 and z < endPos
        inside = in1 + in2 + in3
        return inside

    elif cell == 'gCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.05965
        in2 = r < 0.066-z and z > 0.05965 and z < 0.0635
        in3 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in4 = r < 0.030 and z >= 0.0640 and z <  endPos
        inside = in1 + in2 + in3 + in4
        return inside

    elif cell == 'hCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.05965
        in2 = r < 0.066-z and z > 0.05965 and z < 0.0635
        in3 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in4 = r < z-0.0615 and z > 0.0640 and z < 0.06785
        in5 = r < 0.030 and z >= 0.06785 and z < endPos #Remember to extend endPos!
        inside = in1 + in2 + in3 + in4 + in5

    elif cell == 'jCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.05965
        in2 = r < 0.066-z and z > 0.05965 and z < 0.0635
        in3 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in4 = r < (3.85/7.9)*(z-0.064)+0.0025 and z > 0.0640 and z < 0.0719
        in5 = r < 0.030 and z >= 0.0719 and z < endPos #Remember to extend endPos!
        inside = in1 + in2 + in3 + in4 + in5
        return inside

    elif cell == 'kCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.0624
        in2 = r < (38.5/11)*(0.0624-z)+0.00635 and z > 0.0624 and z < 0.0635
        in3 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in4 = r < (3.85/7.9)*(z-0.064)+0.0025 and z > 0.0640 and z < 0.0719
        in5 = r < 0.030 and z >= 0.0719 and z < endPos #Remember to extend endPos!
        inside = in1 + in2 + in3 + in4 + in5
        return inside

    elif cell == 'mCell':
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

    elif cell == 'nCell':
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

    elif cell == 'pCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.0635
        in2 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in3 = r < 0.009 and z > 0.0640 and z < 0.067
        in4 = r < 0.00635 and z > 0.067 and z < 0.0721
        in5 = r < 0.0025 and z > 0.0721 and z < 0.0726
        in6 = r < 0.030 and z >= 0.0726 and z < endPos
        inside = in1 + in2 + in3 + in4 + in5 + in6
        return inside

    elif cell == 'qCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.0635
        in2a = r < 0.0025 and z > 0.0635 and z < 0.064
        in2b = r < 0.00635 and r > 0.00585 and z > 0.0635 and z < 0.064
        in3 = r < 0.030 and z >= 0.064 and z < endPos
        inside = in1 + in2a + in2b + in3
        return inside

    elif cell == 'rCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.0635
        in2 = r < 0.0025 and z > 0.0635 and z < 0.0640
        in3 = r < 0.030 and z >= 0.0640 and z < endPos
        inside = in1 + in2 + in3
        return inside 
    
    elif cell == 'yCell':
        in1 = r < 0.00635 and z > 0.015 and z < 0.0531
        in2 = r < 0.0025 and z > 0.0531 and z < 0.0536
        in3 = r < 0.030 and z >= 0.0536 and z < endPos
        inside = in1 + in2 + in3
        return inside
    
    elif cell == 'tCell':
        in0 = r < 0.002 and z > 0.015 and z < 0.035
        in1 = r < 0.00635 and z > 0.035 and z < 0.0881
        in2 = r < 0.0025 and z > 0.0881 and z < 0.0886
        in3 = r < 0.030 and z >= 0.0886 and z < endPos
        inside = in1 + in2 + in3
        return inside

    else:
        raise ValueError('Could not find bounds for geometry {}'.format(cell))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Specify simulation params')
    parser.add_argument('-ff', metavar='flows/X_Cell/DS2xyyy.DAT', help='File containing flowfield from DS2V output') # Specify flowfield
    parser.add_argument('-out', metavar='xyyy.dat', help = 'Output file name, to store trajectory data') # Specify output filename

    parser.add_argument('--mult', type=float, dest='mult', action='store', help='Multiplier for the collision cross section') # Specify cross section multiplier (optional)
    parser.add_argument('--npar', type=int, dest='npar', action='store', help='Number of particles to simulate') #Specify number of particles to simulate (optional, defaults to 1)
    parser.add_argument('--lite', dest='lite', action='store_true', help = 'Set TRUE if recording trajectories inside the cell is not necessary')

    parser.add_argument('--init_mode', type=int, dest='init_mode', action='store', help='Code number for initial particle distributions')
    parser.add_argument('--probe_mode', dest='probe_mode', action='store_true', help='Set TRUE if only particles final locations are needed')
    # parser.add_argument('--correc', dest='correc', action='store', help='Specify whether to use *corrected* velocity distribution')
    parser.set_defaults(lite=False, mult=5, npar=1, init_mode=0, probe_mode=False, correc=1) #Defaults to LITE_MODE=False, 1 particle and CROSS_MULT=5
    args = parser.parse_args()

    flowField = args.ff
    outfile = args.out

    NPAR = args.npar
    CROSS_MULT = args.mult
    LITE_MODE = args.lite
    INIT_COND = args.init_mode
    PROBE_MODE = args.probe_mode

    CORRECTION = 1 #CORRECTION=1 sets which distributions/sampling methods are to be used in sampling particle data

    sim = ParticleTracing(flowField, NPAR, CROSS_MULT, LITE_MODE, INIT_COND, PROBE_MODE, CORRECTION)

    print("Particle number {0}, crossmult {1}, LITE_MODE {2}, INIT_COND {3}".format(NPAR, CROSS_MULT, LITE_MODE, INIT_COND))
    print("PROBE_MODE {}".format(PROBE_MODE))

    sim.parallel_main(outfile)
    # results = sim.nonparallel_main()
    # f = open(outfile, 'w+')
    # f.write('x (mm)   y (mm)   z (mm)   vx (m/s)   vy (m/s)   vz (m/s)   time (ms)   dens\n')
    # f.write(''.join(map(str, list(itertools.chain.from_iterable(results)))))
    # f.close()
