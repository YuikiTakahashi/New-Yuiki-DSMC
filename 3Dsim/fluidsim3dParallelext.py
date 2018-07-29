#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:43:30 2018

@author: Dave
"""

import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st
#import plotly as plo
#import plotly.plotly as py
#import plotly.graph_objs as go
import scipy.integrate as integrate
from multiprocessing import Pool
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

# Global variable initialization: these values are irrelevant
vx, vy, vz, xFlow, yFlow, zFlow = 0, 0, 0, 0, 0, 0

def set_derived_quants():
    '''
    This function must be run whenever n, T, T_s, vx, vy, or vz are changed.
    '''
    global U, vMean, vMeanM, coll_freq, dt
#    U = 1.5 * kb * T
    vMean = 2 * (2 * kb * T / (m * np.pi))**0.5
    vMeanM = 2 * (2 * kb * T_s / (M * np.pi))**0.5

    # Calculate average relative velocity given current species vx,vy,vz
    #vRel = integrate.tplquad(lambda p,t,v: np.sqrt((vx-xFlow-v*np.sin(t)*np.cos(p))**2\
     #                                              + (vy-yFlow-v*np.sin(t)*np.sin(p))**2\
      #                                             + (vz-zFlow-v*np.cos(t))**2) * \
    #(m/(2*np.pi*kb*T))**1.5 * 4*np.pi * v**2 * np.exp(-m*v**2/(2*kb*T)) * np.sin(t)/(4*np.pi),\
    #0, vMean*4, lambda v: 0, lambda v: np.pi, lambda v,t: 0, lambda v,t: 2*np.pi)[0]

    coll_freq = n * cross * (vMean + np.sqrt((vx-xFlow)**2 + (vy-yFlow)**2 + (vz-zFlow)**2)/3) # vRel
    dt = 0.01 / coll_freq # ∆t satisfying E[# collisions in 100∆t] = 1.

set_derived_quants()

# Remember to set form-dependent parameters below.


# =============================================================================
# Probability Distribution Functions
# =============================================================================

# Maxwell-Boltzmann Velocity Distribution for ambient molecules
# Used temporarily since high-precision coll_vel_pdf distributions are too slow
class vel_pdf(st.rv_continuous):
    def _pdf(self,x):
        return (m/(2*np.pi*kb*T))**1.5 * 4*np.pi * x**2 * np.exp(-m*x**2/(2*kb*T))
vel_cv = vel_pdf(a=0, b=5*vMean, name='vel_pdf') # vel_cv.rvs()

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

try:
    flowField = np.loadtxt('DS2FF017.dat', skiprows=1) # Assumes only first row isn't data.
except:
    print("Note: No Flow Field DSMC data.")

def dsmcQuant(x0, y0, z0, col):
    '''
    Interpolate quantities from an axially symmetric DSMC FF file.
    Col parameter: 2 = number density, 7 = temperature, 4 = velocities
    '''
    r0 = (x0**2 + y0**2)**0.5
    quants = flowField[:, col]
    if col in [2, 7]:
        np.place(quants, quants==0, 1) # To not have log(0)
        quants = np.log(quants) # Interpolate densities/temps on a log scale
    zs = flowField[:, 0] # DSMC calls these 'x' values
    rs = flowField[:, 1] # DSMC calls these 'y' values
    dists = (zs-z0)**2 + (rs-r0)**2

    # Find 3 nearest (r, z) points to (r0, z0).
    ind1 = np.argmin(dists)
    z1, r1, quant1 = zs[ind1], rs[ind1], quants[ind1]

    dists[ind1] = np.amax(dists)
    ind2 = np.argmin(dists)
    z2, r2, quant2 = zs[ind2], rs[ind2], quants[ind2]

    dists[ind2] = np.amax(dists)
    ind3 = np.argmin(dists)
    z3, r3, quant3 = zs[ind3], rs[ind3], quants[ind3]
    '''
    # Solve quant_i = A * z_i + B * r_i + C for A, B, C using 3 nearest (r_i, z_i).
    # Sometimes the 3 nearest points are collinear; this is handled in 'except'.

    while True:
        if ((r1 == r2 and r2 == r3) or (z1 == z2 and z2 == z3) or z0 in [max([z0, z1, z2, z3]), min([z0, z1, z2, z3])] or r0 in [max([r0, r1, r2, r3]), min([r0, r1, r2, r3])]):
            dists[ind3] = np.amax(dists)
            ind3 = np.argmin(dists)
            z3, r3, quant3 = zs[ind3], rs[ind3], quants[ind3]
        else:
            left = np.array([[z1, r1, 1], [z2, r2, 1], [z3, r3, 1]])
            right = np.array([quant1, quant2, quant3])
            a, b, c = np.linalg.lstsq(left, right)[0]
            break

    quant0 = a * z0 + b * r0 + c
    '''
    quant0 = np.mean([quant1, quant2, quant3])
    if col == 4:
        Vz = quant0
        vr = dsmcQuant(x0, y0, z0, 5)
        vPerpCw = dsmcQuant(x0, y0, z0, 6)

        theta = np.arctan2(y0, x0)
        rot = np.pi/2 - theta
        Vx = np.cos(rot) * vPerpCw + np.sin(rot) * vr
        Vy = -np.sin(rot) * vPerpCw + np.cos(rot) * vr

        return Vx, Vy, Vz

    if col in [2, 7]:
        return np.exp(quant0)

    return quant0

def inBounds(x, y, z, form='box'):
    '''
    Return Boolean value for whether or not a position is within
    the boundary of "form".
    '''
    if form in ['box', 'curvedFlowBox']:
        inside = abs(x) <= 0.05 and abs(y) <= 0.05 and abs(z) <= 0.05
    elif form == 'currentCell':
        r = np.sqrt(x**2+y**2)
        #in1 = r < 0.0015875 and z > 0.001 and z < 0.015
        in2 = r < 0.00635 and z > 0.015 and z < 0.0635
        in3 = r < 0.0025 and z > 0.0635 and z < 0.0750
        inside = in2 + in3
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
        xFlow, yFlow, zFlow = dsmcQuant(x, y, z, 4)
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
        n = dsmcQuant(x, y, z, 2)
        if abs(n) > 1e23:
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
        T = dsmcQuant(x, y, z, 7)
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

def numClose(x, y, z, xs, ys, zs, dist=0.008):
    '''
    Gives the number of particles in xs,ys,zs within a cube of length 'dist'
    around a particle at x, y, z.
    '''
    num_close = -1 # Will always see ≥ 1 close particle (itself)
    for i in range(len(xs)):
        if abs(x-xs[i]) < dist and abs(y-ys[i]) < dist and abs(z-zs[i]) < dist:
            num_close += 1
    return num_close

def collide():
    '''
    For current values of position (giving ambient flow rate) and velocity,
    increment vx, vy, vz according to collision physics.
    '''
    global vx, vy, vz, dt
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

def pathTrace(T_s0=4, form='box'):
    '''
    Track and return a list of sequential positions over time on a given path.
    '''
    global vx, vy, vz, t # Global velocities used for collide() and updating PDFs
    x, y, z = initial_species_position(.01, form)
    vx, vy, vz = initial_species_velocity(T_s0)
    t = 0
    xs = [x]
    ys = [y]
    zs = [z]

    while inBounds(x, y, z, form):
        # Typically takes 5-20 ms to leave box
        updateParams(x, y, z, form)
        if np.random.uniform() < 0.01: # 1/100 chance of collision
            collide()
        t += dt
        x += vx * dt
        y += vy * dt
        z += vz * dt
        xs.append(x)
        ys.append(y)
        zs.append(z)

    print("Time to wall: "+str(t)+" seconds")
    return xs, ys, zs

def getDensityTrend(filename):
    '''
    Record mean time to hit wall for a range of ambient densities.
    '''
    global n
    with open(filename, 'w') as f:
        f.write("n (per cu. m)   mean iterations   mean time to stick   sig iter   sig time\n")
        for density in [10**9, 10**9*5, 10**10, 10**10*5, 10**11, 10**11*5, 10**12]:
            n = density
            lens = []
            times = []
            for j in range(1000):
                xs, ys, zs = pathTrace()
                lens.append(len(xs))
                times.append(t)
            meanLen = str(np.mean(lens))
            meanTime = str(np.mean(times))
            stdLen = str(np.std(lens)/(len(lens)**0.5))
            stdTime = str(np.std(times)/(len(times)**0.5))
            f.write('%.1E'%n+' '+meanLen+' '+meanTime+' '+stdLen+' '+stdTime+'\n')
    f.close()

def getData(t1=0.00005, t2=0.0015, step=0.00005, trials=400, x0=0, y0=0, z0=0, T_s0=4):
    '''
    For a range of total times, record expected values of final position,
    square-distance, and speed near end of path.
    '''
    global vx, vy, vz
    updateParams(x0, y0, z0, 'open')
    with open('_'.join(map(str,[int(1e6*t1), int(1e6*t2), int(T_s0), int(xFlow), \
                                int(M/m), int(np.log10(float(n)))]))+'.dat', 'w') as f:
        f.write('   '.join(['time (s)','xAvg (m)','yAvg (m)', 'zAvg (m)', 'SqrAvg (sq. m)',\
                          'SpeedAvg (m/s)','sigX','sigY', 'sigZ', 'sigSqr','sigSpeed'])+'\n')
        for time in np.arange(t1, t2, step):
            print(time)
            xs = []
            ys = []
            zs = []
            squares = []
            speedAvgs = []
            for j in range(trials):
                print(j)
                t = 0
                x, y, z = x0, y0, z0
                vx, vy, vz = initial_species_velocity(T_s0)
                speeds = []
                while t < time:
                    updateParams(x, y, z, 'open')
                    if np.random.uniform() < 0.01: # 1/100 chance of collision
                        collide()
                    t += dt
                    x += vx * dt
                    y += vy * dt
                    z += vz * dt
                    if t > 0.8 * time:
                        speeds.append((vx**2 + vy**2 + vz**2)**0.5)
                speedAvgs.append(np.mean(speeds))
                squares.append(x**2+y**2+z**2)
                xs.append(x)
                ys.append(y)
                zs.append(z)
            meanx = str(np.mean(xs))
            meany = str(np.mean(ys))
            meanz = str(np.mean(zs))
            meanSq = str(np.mean(squares))
            meanSpeed = str(np.mean(speedAvgs))
            stdx = str(np.std(xs)/(len(xs)**0.5))
            stdy = str(np.std(ys)/(len(ys)**0.5))
            stdz = str(np.std(zs)/(len(zs)**0.5))
            stdSq = str(np.std(squares)/(len(squares)**0.5))
            stdSpeed = str(np.std(speedAvgs)/(len(speedAvgs)**0.5))
            f.write(' '.join([str(time),meanx,meany,meanz,meanSq,meanSpeed,\
                              stdx,stdy,stdz,stdSq,stdSpeed])+'\n')
    f.close()

def endPosition(T_s0):
    '''
    Return the final position of a particle somewhere on the (10cm)^3 box.
    Initial position determined randomly in a cube around the origin.
    '''
    global vx, vy, vz
    np.random.seed()
    x, y, z = initial_species_position(.01, 'currentCell')
    vx, vy, vz = initial_species_velocity(T_s0)

    while inBounds(x, y, z, 'currentCell'):
        # Typically takes few ms to leave box
        updateParams(x, y, z, 'currentCell')
        if np.random.uniform() < 0.01: # 1/100 chance of collision
            collide()
        x += vx * dt
        y += vy * dt
        z += vz * dt
    if z > 0.075:
        # Linearly backtrack to aperture
        z = 0.075
        x -= (z-0.075)/(vz * dt) * (vx * dt)
        y -= (z-0.075)/(vz * dt) * (vy * dt)
    return x, y, z, np.sqrt(vx**2+vy**2), vz


# =============================================================================
# Image-generating functions
# =============================================================================
"""
def showPaths(filename, form='currentCell'):
    '''
    Generate a scatter plot of 1000 equally t-spaced points from each of
    three independent paths generated by pathTrace().
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for j in range(1):
        xs, ys, zs = pathTrace(form=form)
        for i in range(0, len(xs)):
            if j == 1 :
                colour = plt.cm.Greens(int(264. * i / len(xs)))
            elif j == 2:
                colour = plt.cm.Blues(int(264. * i / len(xs)))
            else:
                colour = plt.cm.Reds(int(264. * i / len(xs)))
            ax.scatter(xs[i], ys[i], zs[i], s=.5, c=colour)

    ax.set_xlim(-.01, .01)
    ax.set_ylim(-.01, .01)
    ax.set_zlim(0, .05)
    plt.title("Paths of a Heavy Particle, n = %.0E" %n)
    ax.set_xlabel('x, meters')
    ax.set_ylabel('y, meters')
    ax.set_zlabel('z, meters')
"""
def showWalls(filename, form='currentCell'):
    '''
    Generate a scatter plot of final positions of molecules as determined by
    the endPosition function parameters.
    '''
    with open("finalPositions%s.dat"%filename.strip('CellWalls.jpg'), "w") as f:
        inputs = np.ones(10000)*4
        with Pool(processes=50) as pool:
            result = pool.map(endPosition, inputs, 200)
        f.write('\n'.join(map(str, result)).replace(")", "").replace(",", "").replace("(", ""))
    f.close()
"""
def pointEmitter(filename="ConstantFlowEmitter.dat", x0=0, y0=0, z0=0, T_s0=4, form='box'):
    '''
    Assuming constant-rate emission of species molecules of random velocities
    (determined by a T = T_s0 thermal distribution) from (x0, y0, z0), record
    the positions of all emitted particles after some time, assign each
    particle a density, and plot them with density-scaling color values.
    '''
    global vx, vy, vz
    xs, ys, zs = [], [], []
    for time in np.arange(0.01, 6, 0.01): # ms
        print(time)
        x, y, z = x0, y0, z0
        vx, vy, vz = initial_species_velocity(T_s0)
        t = 0
        while t < time * 1e-3:
            updateParams(x, y, z, form)
            if np.random.uniform() < 0.01: # 1/100 chance of collision
                collide()
            t += dt
            x += vx * dt
            y += vy * dt
            z += vz * dt
        xs.append(x)
        ys.append(y)
        zs.append(z)

    densities = []
    for i in range(len(xs)):
        densities.append(numClose(xs[i], ys[i], zs[i], xs, ys, zs))

    trace = go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', \
        marker=dict(size=12,color=densities,colorscale='Rainbow',opacity=0.8))
    fig = go.Figure(data=[trace], layout=go.Layout())
    plo.offline.plot(fig, filename='simple-3d-scatter')

    with open(filename, 'w') as f:
        for i in range(len(xs)):
            f.write(str(xs[i])+' '+str(ys[i])+' '+str(zs[i])+'\n')
    f.close()
"""
showWalls('017extCellWalls.jpg')
#flowField = np.loadtxt('DS2FFdenser.DAT', skiprows=1)
#showWalls('denserCellWalls.jpg')
#flowField = np.loadtxt('DS2FFspeeder.DAT', skiprows=1)
#showWalls('speederCellWalls.jpg')
# Play around with initial position and velocity
# Explore different DSMC simulations
