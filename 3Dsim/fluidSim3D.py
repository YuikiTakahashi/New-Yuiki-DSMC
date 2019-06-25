#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:43:30 2018

@author: Dave
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.interpolate as si
import plotly as plo
import plotly.graph_objs as go
import scipy.integrate as integrate

# =============================================================================
# Constant Initialization
# =============================================================================

kb = 1.38 * 10**-23
NA = 6.022 * 10**23
sccmSI = 7.45E-07 * NA
T = 4 # Ambient temperature (K)
T_s = 4 # Species temperature (K) for initial velocity distributions
m = 0.004 / NA # Ambient gas mass (kg)
M = .190061 / NA # Species mass (kg)
massParam = 2 * m / (m + M)
n = 10**21 # m^-3
crossBB = 4 * np.pi * (140 * 10**(-12))**2 # two-helium cross sectional area
cross = 4*crossBB # Rough estimate of He-YbOH cross sectional area
cross *= 5 # Manual

# Global variable initialization: these values are irrelevant
vx, vy, vz, xFlow, yFlow, zFlow = 0, 0, 0, 0, 0, 0

def set_derived_quants():
    '''
    This function must be run whenever n, T, T_s, vx, vy, or vz are changed.
    '''
    global U, vMean, vMeanM, coll_freq, dt, no_collide
#    U = 1.5 * kb * T
    vMean = 2 * (2 * kb * T / (m * np.pi))**0.5
    vMeanM = 2 * (2 * kb * T_s / (M * np.pi))**0.5

    # Calculate average relative velocity given current species vx,vy,vz
#    vRel = integrate.tplquad(lambda p,t,v: np.sqrt((vx-xFlow-v*np.sin(t)*np.cos(p))**2\
#                                                   + (vy-yFlow-v*np.sin(t)*np.sin(p))**2\
#                                                   + (vz-zFlow-v*np.cos(t))**2) * \
#    (m/(2*np.pi*kb*T))**1.5 * 4*np.pi * v**2 * np.exp(-m*v**2/(2*kb*T)) * np.sin(t)/(4*np.pi),\
#    0, vMean*4, lambda v: 0, lambda v: np.pi, lambda v,t: 0, lambda v,t: 2*np.pi)[0]

    coll_freq = n * cross * vMean # vRel
    if 0.1 / coll_freq < 1e-5:
        dt = 0.1 / coll_freq # ∆t satisfying E[# collisions in 100∆t] = 1.
        no_collide = False
    else: # Density is so low that collision frequency is near 0
        no_collide = True # Just don't collide.

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

# =============================================================================
# Must have axis-of-symmetry "wall" data in FF for this to work.
# =============================================================================
try:
    raise # Skip data loading when only want file analysis function
    flowField = np.loadtxt('flows/DS2FF019.DAT', skiprows=1) # Assumes only first row isn't data.
    zs, rs, dens, temps = flowField[:, 0], flowField[:, 1], flowField[:, 2], flowField[:, 7]
    vzs, vrs, vps = flowField[:, 4], flowField[:, 5], flowField[:, 6]
    quantHolder = [zs, rs, dens, temps, vzs, vrs, vps]
    grid_x, grid_y = np.mgrid[0.010:0.12:4500j, 0:0.030:1500j] # high density, to be safe.
    # The high grid density increases overhead time for running this initialization,
    # but the time for running particle simulations is unaffected.
    grid_dens = si.griddata(np.transpose([zs, rs]), np.log(dens), (grid_x, grid_y), 'nearest')
    grid_temps = si.griddata(np.transpose([zs, rs]), temps, (grid_x, grid_y), 'nearest')
    grid_vzs = si.griddata(np.transpose([zs, rs]), vzs, (grid_x, grid_y), 'nearest')
    grid_vrs = si.griddata(np.transpose([zs, rs]), vrs, (grid_x, grid_y), 'nearest')
    grid_vps = si.griddata(np.transpose([zs, rs]), vps, (grid_x, grid_y), 'nearest')
    # The 'nearest' gridding has not been thoroughly tested for edge-case accuracy.
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
    global vx, vy, vz
    Theta = Theta_cv.rvs()
    Phi = np.random.uniform(0, 2*np.pi)
    vx_amb, vy_amb, vz_amb = getAmbientVelocity(simple=True)
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
        if np.random.uniform() < 0.01 and no_collide == False: # 1/100 chance of collision
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

def getCrossTrend():
    global cross
    for i in [0.5, 1]:
        cross = cross * i
        set_derived_quants()
        ts = []
        for j in range(50):
            x, y, z, t = 0, 0, 0, 0
            vx, vy, vz = initial_species_velocity(T_s0=4)

            while inBounds(x, y, z):
                if np.random.uniform() < 0.1 and no_collide==False:
                    collide()
                x += vx * dt
                y += vy * dt
                z += vz * dt
                t += dt
            ts.append(t)

        print(i, np.mean(ts), np.std(ts)/np.sqrt(50))

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
                    if np.random.uniform() < 0.01 and no_collide == False: # 1/100 chance of collision
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

def endPosition(extPos=0.12):
    '''
    Return the final position of a particle somewhere in the cell or else
    past the aperture within a distance extPos.
    '''
    global vx, vy, vz
    trajectory = open('trajectory.dat', 'w')
    np.random.seed()
    x, y, z = initial_species_position(.01, 'currentCell')
    vx, vy, vz = initial_species_velocity(T_s0=4)
    inCell = True
    xAp, yAp, zAp, vzAp, vrAp = 0, 0, 0, 0, 0
    trajectory.write(' '.join(map(str, [round(x,7), round(y,7), round(z,7), \
                                        round(np.sqrt(vx**2+vy**2),1), round(vz,1)]))+'\n')
    while inBounds(x, y, z, 'currentCell', extPos):
        # Typically takes few ms to leave box
        updateParams(x, y, z, 'currentCell')
        if np.random.uniform() < 0.1 and no_collide==False: # 1/10 chance of collision
            collide()
            trajectory.write(' '.join(map(str, [round(x,7), round(y,7), round(z,7), \
                                        round(np.sqrt(vx**2+vy**2),1), round(vz,1)]))+'\n')
        x += vx * dt
        y += vy * dt
        z += vz * dt
        if z > 0.064 and inCell == True:
            # Record properties at aperture
            zAp = 0.064
            xAp = x - (z-0.064)/(vz * dt) * (vx * dt)
            yAp = y - (z-0.064)/(vz * dt) * (vy * dt)
            vzAp, vrAp = vz, np.sqrt(vx**2+vy**2)
            inCell = False

    if z > extPos:
        # Linearly backtrack to boundary
        z = extPos
        x -= (z-extPos)/(vz * dt) * (vx * dt)
        y -= (z-extPos)/(vz * dt) * (vy * dt)
    return xAp, yAp, zAp, vrAp, vzAp, x, y, z, np.sqrt(vx**2+vy**2), vz

# =============================================================================
# Image-generating functions
# =============================================================================

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
    plt.savefig(filename)
    plt.show()

def showWalls(filename, form='currentCell', write_vels=False):
    '''
    Generate a scatter plot of final positions of molecules as determined by
    the endPosition function parameters.
    '''
    if write_vels == True:
        fv = open('apertureVels%s.dat'%filename[:-13], 'w')
        fv.write("Radial Velocity   Axial Velocity\n")
    with open("finalPositions%s.dat"%filename[:-13], "w") as f:
        for j in range(1):
            print(j)
            x, y, z = endPosition(form=form)
            f.write("%.5f %.5f %.5f \n"%(x, y, z))
            if write_vels == True and z == 0.064 and np.sqrt(x**2+y**2) < 0.00075:
                fv.write("%.5f %.5f \n"%((vx**2+vy**2)**0.5, vz))
            colour = plt.cm.Greens(int(64 + 200 * abs(y) / 0.05)) # color = y coord
            if form not in ['currentCell']:
                plt.plot (x, z, c=colour, marker='+', ms=13)
            else:
                plt.plot(z, np.sqrt(x**2+y**2), c=colour, marker='+', ms=13)
    f.close()
    if form == 'currentCell':
        plt.vlines(0.001, 0, 0.0015875, colors='gray', linewidths=.5)
        plt.hlines(0.0015875, 0.001, 0.015, colors='gray', linewidths=.5)
        plt.vlines(0.015, 0.0015875, 0.00635, colors='gray', linewidths=.5)
        plt.hlines(0.00635, 0.015, 0.0635, colors='gray', linewidths=.5)
        plt.vlines(0.0635, 0.00635, 0.0025, colors='gray', linewidths=.5)
        plt.hlines(0.0025, 0.0635, 0.064, colors='gray', linewidths=.5)
        plt.vlines(0.064, 0.0025, 0.009, colors='gray', linewidths=.5)
        plt.hlines(0.009, 0, 0.064, colors='gray', linewidths=.5)
        plt.xlim(0, 0.07)
        plt.ylim(0, 0.01)
    else:
        plt.hlines(0.05, -.05, .05, colors='gray', linewidths=.5)
        plt.hlines(-0.05, -.05, .05, colors='gray', linewidths=.5)
        plt.vlines(0.05, -.05, .05, colors='gray', linewidths=.5)
        plt.vlines(-0.05, -.05, .05, colors='gray', linewidths=.5)
        plt.xlim(-.055, .055)
        plt.ylim(-.055, .055)
    plt.title("Final Positions of a Heavy Particle")
    plt.xlabel('z, meters')
    plt.ylabel('r, meters')
    plt.savefig(filename)
    plt.show()

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
            if np.random.uniform() < 0.01 and no_collide == False: # 1/100 chance of collision
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

# =============================================================================
# Data file analysis
# =============================================================================

def analyzeWallData(file_ext, pos):
    '''
    Running a Parallel fluid sim script produces a file with ten columns;
    five for positions and velocities at the aperture, and
    five for positions and velocities at a z-value ("pos") beyond the aperture.
    This function produces a graph of the end positions on the walls and
    prints the number of molecules making it to the aperture and to "pos".
    It then plots position/velocity distributions at aperture and "pos".
    '''
    f = np.loadtxt('/Users/Dave/Documents/2018 SURF/3Dsim/Data/%s.dat'%file_ext)

    xs = np.array(f[:, 5])
    ys = f[:, 6]
    zs = f[:, 7]
    colour = plt.cm.Greens(100) # color = y coord
    plt.plot(zs, np.sqrt(xs**2+ys**2), '+', c=colour, ms=13)
    plt.vlines(0.001, 0, 0.0015875, colors='gray', linewidths=.5)
    plt.hlines(0.0015875, 0.001, 0.015, colors='gray', linewidths=.5)
    plt.vlines(0.015, 0.0015875, 0.00635, colors='gray', linewidths=.5)
    plt.hlines(0.00635, 0.015, 0.0635, colors='gray', linewidths=.5)
    plt.vlines(0.0635, 0.00635, 0.0025, colors='gray', linewidths=.5)
    plt.hlines(0.0025, 0.0635, 0.064, colors='gray', linewidths=.5)
    plt.vlines(0.064, 0.0025, 0.009, colors='gray', linewidths=.5)
    plt.hlines(0.009, 0, 0.064, colors='gray', linewidths=.5)
    plt.xlim(0, 0.1)
    plt.ylim(0, 0.01)
    plt.show()

    unique, counts = np.unique(f[:, 2], return_counts=True)
    numAp = counts[unique.tolist().index(0.064)]
    unique, counts = np.unique(f[:, 7], return_counts=True)
    numPost = counts[unique.tolist().index(pos)]
    print('%d/10,000 (%.1f%%) made it to the aperture.'%(numAp, numAp/100.))


    fAp = f[f[:, 2]==0.064]
    xs, ys = fAp[:, 0], fAp[:, 1]
    plt.plot(xs, ys, '.')
    plt.xlabel('x, meters')
    plt.ylabel('y, meters')
    plt.title("Radial Positions at the Aperture")
    plt.tight_layout()
    plt.savefig('images/'+file_ext+'PosAp.png')
    plt.clf()

    vrs, vzs = fAp[:, 3], fAp[:, 4]
    plt.plot(vrs, vzs, '.')
    plt.title("Velocity Distribution at the Aperture")
    plt.ylabel('Axial velocity, m/s')
    plt.xlabel('Radial velocity, m/s')
    plt.tight_layout()
    plt.savefig('images/'+file_ext+'VelAp.png')
    plt.clf()
    plt.hist(vzs, bins=15)
    plt.xlabel('Axial velocity, m/s')
    plt.ylabel('Frequency')
    plt.savefig('images/hist.png')

    print('Radial velocity at aperture: %.1f +- %.1f m/s'\
          %(np.mean(vrs), np.std(vrs)))
    print('Axial velocity at aperture: %.1f +- %.1f m/s'\
          %(np.mean(vzs), np.std(vzs)))
    print('Angular spread at aperture: %.1f deg \n'\
          %(180/np.pi * 2 * np.arctan(np.mean(vrs)/2/np.mean(vzs))))


    print('%d/10,000 (%.1f%%) made it to z = %.3f m.'%(numPost, numPost/100., pos))
    plt.clf()
    fPost = f[f[:, 7]==pos]
    xs, ys = fPost[:, 5], fPost[:, 6]
    plt.plot(xs, ys, '.')
    plt.xlabel('x, meters')
    plt.ylabel('y, meters')
    plt.title("Radial Positions %.1f cm past the Aperture"%((pos-0.064)*100))
    plt.tight_layout()
    plt.savefig('images/'+file_ext+'PosPost.png')

    plt.clf()
    vrs, vzs = fPost[:, 8], fPost[:, 9]
    plt.plot(vrs, vzs, '.')
    plt.title("Velocity Distribution %.1f cm past the Aperture"%((pos-0.064)*100))
    plt.ylabel('Axial velocity, m/s')
    plt.xlabel('Radial velocity, m/s')
    plt.tight_layout()
    plt.savefig('images/'+file_ext+'VelPost.png')

    print('Radial velocity %.1f cm past the aperture: %.1f +- %.1f m/s'\
          %((pos-0.064)*100, np.mean(vrs), np.std(vrs)))
    print('Axial velocity %.1f cm past the aperture: %.1f +- %.1f m/s'\
          %((pos-0.064)*100, np.mean(vzs), np.std(vzs)))
    print('Angular spread %.1f cm past the aperture: %.1f deg'\
          %((pos-0.064)*100, \
            180/np.pi * 2 * np.arctan(np.mean(vrs)/np.mean(vzs))))

##########################******************************#########################################


def analyzeTrajData(file_ext, write_file=None, pos=0.064, write=False, plots=False,rad_mode=False, dome_rad=0.02):
    '''
    Running a Parallel open trajectory script produces a file with six columns;
    three each for positions and velocities.
    This function produces a graph of the end positions on the walls and
    prints the number of molecules making it to the z-value "pos", if rad_mode
    is False, or making it to the dome with radius dome_rad, if rad_mode is True.
    In other words, rad_mode switches the final surface to analyze particles
    between xy-planes and domes.
    Plots position/velocity distributions at selected analysis surface.
    The default xy-plane is set to the aperture position, z=0.064 m.
    The default dome is set to radius r=0.02 m.
    '''
    print('The aperture is at z = 0.064 m.')

    #Coordinates of bowl center
    x_center=0
    y_center=0
    z_center=64 #64 mm

    pos0 = pos
    pos *= 1000 # trajectory file data is in mm

    if rad_mode == False:
        print('Analysis of data for z = %g m, equal to %g m past the aperture:'%(pos0, pos0-0.064))
        dome_rad = pos - z_center
        print("dome_rad is equal to {0}".format(dome_rad))

    elif rad_mode == True:
        #Estimating the freezing point to be at 0.02 m past aperture
        dome_rad0 = dome_rad
        dome_rad *= 1000 #file data is in mm
        print('Analysis of data at dome r = %g m, centered at aperture:'%dome_rad0)

    #Nx6 array. Organized by x,y,z,vx,vy,vz
    f = np.loadtxt('/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/TimeColumn/%s.dat'%file_ext, skiprows=1)
    #f = np.loadtxt('/Users/gabri/Desktop/HutzlerSims/Gas-Simulation/3Dsim/Data/%s.dat'%file_ext, skiprows=1)

    flowrate = {'traj017d':5, 'traj018':20, 'traj019':50, 'traj020':10, 'traj021':2,\
                'traj022':100, 'traj023':200,'flow_17':5,'flow_18_a':20,\
                'flow_19_a':50,'flow_20':10,'flow_21':2,'flow_22':100,\
                'lite10':5, 'f17_lite':5, 'f18_lite':20, 'f19_lite':50,\
                'f20_lite':10, 'f21_lite':2, 'f22_lite':100, 'f23_lite':200}[file_ext]


    num = 0 #number of simulated particles
    for i in range(len(f)):
        if not np.any(f[i]):
            num += 1 # count number of particles simulated in file

    #Adding a 7th, 8th, 9th entries to array for times, radii, thetas
    #Output pdata will look like:
    #[x, y, z, vx, vy, vz, t, r, theta]
    #Recall that the output from simulation only comes with the first 7 of these

    finals = np.zeros((num, 9))
    j = 0
    for i in range(len(f)):
        if (not np.any(f[i])) and f[i-1][2] != 120 and np.sqrt(f[i-1][0]**2 + f[i-1][1]**2) < 30:
            #Finds final row of data for adequate particles. Not condensed, but those that
            #went past aperture. Maybe condensed in vacuum chamber?
            x, y, z, vx, vy, vz, tim = f[i-1]
            r = np.sqrt((x_center-x)**2+(y_center-y)**2+(z_center-z)**2)
            theta = (180/np.pi) * np.arccos((z-z_center)/r)
            finals[j] = np.array([x, y, z, vx, vy, vz, tim, r, theta])
            j += 1

    found = False
    for i in range(len(f)):
        #If still looking and z coordinate is past the query boundary pos
        if found == False and f[i][2] >= pos and rad_mode == False:
            #Linearly backtrack to boundary at pos
            x, y, z, vx, vy, vz, tim = f[i-1]
            delta_t = (pos-z)/vz #Negative value
            x += vx * delta_t
            y += vy * delta_t
            tim += delta_t
            r = np.sqrt((x_center-x)**2+(y_center-y)**2+(z_center-z)**2)
            theta = (180/np.pi) * np.arccos((z-z_center)/r)
            finals[j] = np.array([x, y, pos, vx, vy, vz, tim, r, theta])
            j += 1
            found = True

        elif found == False and f[i][2] >= pos and rad_mode == True:

            past_rad = np.sqrt((f[i][0] - x_center)**2 + (f[i][1] - y_center)**2 + (f[i][2] - z_center)**2)
            #print("Dx=%6.4f, Dy=%6.4f, Dz=%6.4f"%(dx,dy,dz))
            #print("Radius past: %6.3f"%past_rad)
            if past_rad >= dome_rad:

                x, y, z, vx, vy, vz, tim = f[i-1]

                dx0 = x - x_center
                dy0 = y - y_center
                dz0 = z - z_center

                r0 = np.sqrt(dx0**2 + dy0**2 + dz0**2)
                vr = (dx0*vx + dy0*vy + dz0*vz) / r0

                # print("First (dx0,dy0,dz0) = ({0}, {1}, {2}), first radius = {3}, first time = {4}".format(dx0,dy0,dz0,r0,tim))
                # print("\n")
                delta_t = (dome_rad - r0) / vr
                dx = dx0 + vx * delta_t
                dy = dy0 + vy * delta_t
                dz = dz0 + vz * delta_t

                tim += delta_t
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                theta = (180/np.pi) * np.arccos(dz/r)

                #print("New (dx,dy,dz) = ({0}, {1}, {2}), new radius = {3}, new time = {4}".format(dx,dy,dz,r,tim))
                #print("\n\n")

                # phi = np.arctan(dy/dx)
                # dx = dome_rad * np.sin(theta) * np.cos(phi)
                # dy = dome_rad * np.sin(theta) * np.sin(phi)
                # dz = dome_rad * np.cos(theta)

                x = x_center + dx
                y = y_center + dy
                z = z_center + dz

                finals[j] = np.array([x, y, z, vx, vy, vz, tim, dome_rad, theta])
                j += 1
                found = True

        elif not np.any(f[i]):
            found = False

    if plots == True:
        xs = finals[:, 0] / 1000.
        ys = finals[:, 1] / 1000.
        zs = finals[:, 2] / 1000.
        colour = plt.cm.Greens(100)
        plt.plot(zs, np.sqrt(xs**2+ys**2), '+', c=colour, ms=13)
        plt.vlines(0.001, 0, 0.0015875, colors='gray', linewidths=.5)
        plt.hlines(0.0015875, 0.001, 0.015, colors='gray', linewidths=.5)
        plt.vlines(0.015, 0.0015875, 0.00635, colors='gray', linewidths=.5)
        plt.hlines(0.00635, 0.015, 0.0635, colors='gray', linewidths=.5)
        plt.vlines(0.0635, 0.00635, 0.0025, colors='gray', linewidths=.5)
        plt.hlines(0.0025, 0.0635, 0.064, colors='gray', linewidths=.5)
        plt.vlines(0.064, 0.0025, 0.009, colors='gray', linewidths=.5)
        plt.hlines(0.009, 0, 0.064, colors='gray', linewidths=.5)
        plt.xlim(0, pos0+0.01)
        plt.show()

    #If analyzing an XY plane at z=pos
    if rad_mode == False:
        unique, counts = np.unique(finals[:, 2], return_counts=True)
        numArrived = counts[unique.tolist().index(pos)]
        pdata = finals[finals[:, 2]==pos] #choose lines with z = pos

    #If analyzing a *dome* with radius dome_rad centered at aperture
    elif rad_mode == True:
        unique, counts = np.unique(finals[:, 7], return_counts=True)
        numArrived = counts[unique.tolist().index(dome_rad)]
        pdata = finals[finals[:, 7]==dome_rad] #choose lines with r = dome_rad


    #xs, ys, zs come in mm, times come in mm/s, velocities come in m/s
    #Divide by a 1000 to convert
    xs, ys, zs, vxs, vys, vzs, times, thetas = pdata[:,0]/1000., pdata[:,1]/1000., pdata[:,2]/1000., \
                                pdata[:,3], pdata[:,4], pdata[:,5], pdata[:,6], pdata[:,8]
#    rs = np.sqrt(xs**2 + ys**2)

    print("Number arrived = {0}, size of xs = {1}".format(numArrived, xs.shape))

    vrs = np.sqrt(vxs**2 + vys**2)
    rs = np.sqrt(xs**2+ys**2)
    #thetas = (180/np.pi) * np.arccos((zs-z_center)*(1/dome_rad))
    phis = (180/np.pi) * np.arctan(ys/xs)

    #Title dependent on whether we analyze plane or dome, and flowrate of DSMC
    if rad_mode == True:
        dep_title = " at r = {0} m".format(dome_rad0)
    else:
        dep_title = " at z = {0} m".format(pos0)

    dep_title_flow = dep_title + "\nFlowrate = {0}".format(flowrate)

    if plots == True:
        plt.plot(100*xs, 100*ys, '.')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.title("Radial Scatter" + dep_title_flow)
        #plt.title("Radial Positions at r = %g m"%dome_rad)
        plt.tight_layout()
#        plt.savefig("/Users/gabri/Desktop/HutzlerSims/Plots/"+file_ext+"/Dome/radial_scatter.png")
        plt.show()
#        plt.savefig('images/'+file_ext+'Pos%g.png')%pos
        plt.clf()

        plt.title("Radial Distribution" + dep_title_flow)
        #plt.title("Radial Distribution at r = %g m"%dome_rad)
        plt.hist(rs,bins=20)
        plt.xlabel('Radius (m)')
        plt.ylabel('Frequency')
#        plt.savefig("/Users/gabri/Desktop/HutzlerSims/Plots/"+file_ext+"/Dome/radial_dist.png")
        plt.show()
        plt.clf()


        plt.plot(vrs, vzs, '.')
        plt.title("Velocity Distribution" + dep_title_flow)
        plt.ylabel('Axial velocity (m/s)')
        plt.xlabel('Radial velocity (m/s)')
        plt.tight_layout()
#        plt.savefig("/Users/gabri/Desktop/HutzlerSims/Plots/"+file_ext+"/Dome/vel_scatter.png")
        plt.show()
#        plt.savefig('images/'+file_ext+'Vel%g.png'%pos)
        plt.clf()

        plt.title("Axial Velocity Distribution" + dep_title_flow)
        plt.hist(vzs, bins=20, range=[0,130])
        plt.xlabel('Axial velocity (m/s)')
        plt.ylabel('Frequency')
#        plt.savefig("/Users/gabri/Desktop/HutzlerSims/Plots/"+file_ext+"/Dome/axial_vel.png")
        plt.show()
#        plt.savefig('images/hist.png')
        plt.clf()

        plt.title("Radial Velocity Distribution" + dep_title_flow)
        plt.hist(vrs, bins=20)
        plt.xlabel('Radial velocity (m/s)')
        plt.ylabel('Frequency')
     #   plt.savefig("/Users/gabri/Desktop/HutzlerSims/Plots/"+file_ext+"/Dome/radial_vel.png")
        plt.show()
        plt.clf()

        plt.title("Theta Distribution" + dep_title_flow)
        plt.hist(thetas, bins=20)
        plt.xlabel("Theta (deg)")
        plt.ylabel("Frequency")
    #    plt.savefig("/Users/gabri/Desktop/HutzlerSims/Plots/"+file_ext+"/Dome/theta.png")
        plt.show()
        plt.clf()

        plt.title("Phi Distribution" + dep_title_flow)
        plt.hist(phis, bins=20)
        plt.xlabel("Phi (deg)")
        plt.ylabel("Frequency")
   #     plt.savefig("/Users/gabri/Desktop/HutzlerSims/Plots/"+file_ext+"/Dome/phi.png")
        plt.show()
        plt.clf()

        plt.title("Arrival Time Distribution"+ dep_title_flow)
        plt.hist(times, bins=20)
        plt.xlabel("Arrival times (ms)")
        plt.ylabel("Frequency")
        plt.show()
        plt.clf()


    stdArrived = np.sqrt(float(numArrived)*(num-numArrived)/num)/num
    spread = 180/np.pi * 2 * np.arctan(np.mean(vrs)/np.mean(vzs))
    spreadB = 180/np.pi * 2 * np.arctan(np.std(vrs)/np.mean(vzs))
    gamma = cross * flowrate * sccmSI / (0.05 * vMean)

    #Estimating d_aperture = 0.0025
    reynolds = 8.0*np.sqrt(2.0) * crossBB * flowrate* sccmSI / (0.0025 * vMean)

    if rad_mode == False:
        print('Analysis of data for z = %g m, equal to %g m past the aperture:'%(pos0, pos0-0.064))

    elif rad_mode == True:
        print('Analysis of data at dome r = %g m, centered at aperture:'%dome_rad0)

    print('%d/%d (%.1f%%) made it to z = %g m.'%(numArrived, num, 100*float(numArrived)/num, pos0))
    print('Standard deviation in extraction: %.1f%%.'%(100*stdArrived))
    print('Radial velocity' + dep_title+ ': %.1f +- %.1f m/s'\
          %(np.mean(vrs), np.std(vrs)))
    print('Axial velocity' + dep_title+ ': %.1f +- %.1f m/s'\
          %(np.mean(vzs), np.std(vzs)))
    print('Angular spread' + dep_title+ ': %.1f deg \n'\
          %(spread))

    print('Theta dist' + dep_title+ ': %.1f +- %.1f deg'\
          %(np.mean(thetas), np.std(thetas)))
    print('Pumpout time dist' + dep_title+ ': %.1f +- %.1f ms \n'\
          %(np.mean(times), np.std(times)))

    # return flowrate, dome_rad, vradMean, vradSD, axMean, axSD, spread, thetMean, thetSD, tMean, tSD

    #David's file format
    # if write == 1:
    #     with open('data/TrajComparisons.dat', 'a') as tc:
    #         tc.write('{:<8g}{:<7d}{:<8.3f}{:<13.3f}{:<8.3f}{:<7.1f}{:<8.1f}{:<7.1f}{:<8.1f}{:<1.1f}\n'\
    #                  .format(pos0, flowrate, gamma, float(numArrived)/num, stdArrived, np.mean(vrs),\
    #                   np.std(vrs), np.mean(vzs), np.std(vzs), spread))
    #     tc.close()

    if write == 1 and rad_mode==True:
        with open('/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/{}'.format(write_file), 'a') as tc:
            tc.write('  '.join(map(str, [dome_rad0, round(float(flowrate),2), round(gamma,3), round(float(numArrived)/num,3),\
                     round(stdArrived,3), round(np.mean(vrs),3), round(np.std(vrs),3), round(np.mean(vzs),3),\
                     round(np.std(vzs),3), round(spread,3), round(np.mean(thetas),3), round(np.std(thetas),3),\
                     round(np.mean(times),3), round(np.std(times),3), round(reynolds,2), round(spreadB,3)] ))+'\n')

        tc.close()

    #These two are identical except for reporting the dome radius versus the plane distance
    elif write == 1 and rad_mode==False:
        with open('/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/{}'.format(write_file), 'a') as tc:
            tc.write('  '.join(map(str, [pos0, round(float(flowrate),2), round(gamma,3), round(float(numArrived)/num,3),\
                     round(stdArrived,3), round(np.mean(vrs),3), round(np.std(vrs),3), round(np.mean(vzs),3),\
                     round(np.std(vzs),3), round(spread,3), round(np.mean(thetas),3), round(np.std(thetas),3),\
                     round(np.mean(times),3), round(np.std(times),3), round(reynolds,2), round(spreadB,3)] ))+'\n')

        tc.close()


#Important: specifying radius also specifies labels for plots
def multiFlowAnalyzeDome(in_file, out_file, radius=0.04, write=False, plot=False):
    fileList = ['f17_lite', 'f18_lite', 'f19_lite', 'f20_lite', 'f21_lite', 'f22_lite', 'f23_lite']

    if write==True:
        for f in fileList:
            analyzeTrajData(f, out_file, write=True, rad_mode=True, dome_rad=radius)

    if plot == True:
        f = np.loadtxt('/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/{}'.format(in_file), skiprows=1)

        rs, frs, gammas, ext, sigE, vR, vRSig, vz, vzSig, spreads,\
        thetas, thetaSig, times, timeSig, reyn, spreadB = f[:,0], f[:,1], f[:,2], \
        f[:,3], f[:,4], f[:,5], f[:,6], f[:,7], f[:,8], f[:,9], \
        f[:,10], f[:,11], f[:,12], f[:,13], f[:,14], f[:,15]

        print("Rs: {},\n frs: {},\n gammas: {},\n times: {}".format(rs,frs,gammas,times))

        plt.title("Pumpout time vs flowrate")
        plt.errorbar(x=frs, y=times, yerr=timeSig, fmt='ro')
        plt.ylabel("Arrival time at r={} m(ms)".format(radius))
        plt.xlabel("Flowrate (SCCM)")
        plt.show()
        plt.clf()

# =============================================================================
# Important: specifing plane also specifies labels for plots. This means it is
# best to keep each data file with the same plane position i.e. keep only a single
# value for the zs
# =============================================================================
def multiFlowAnalyzePlane(file, plane=0.064, write=False, plot=False):
    fileList = ['f17_lite', 'f18_lite', 'f19_lite', 'f20_lite', 'f21_lite', 'f22_lite', 'f23_lite']

    if write==True:
        for f in fileList:
            analyzeTrajData(f, file, pos=plane, write=True, rad_mode=False)

    if plot == True:
        f = np.loadtxt('/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/{}'.format(file), skiprows=1)

        zs, frs, gammas, ext, sigE, vR, vRSig, vz, vzSig, spreads,\
        thetas, thetaSig, times, timeSig, reyn, spreadB = f[:,0], f[:,1], f[:,2], \
        f[:,3], f[:,4], f[:,5], f[:,6], f[:,7], f[:,8], f[:,9], \
        f[:,10], f[:,11], f[:,12], f[:,13], f[:,14], f[:,15]

        print("Zs: {},\n frs: {},\n gammas: {},\n times: {}".format(zs,frs,gammas,times))

        plt.title("Pumpout Time vs Flowrate")
        plt.errorbar(x=frs, y=times, yerr=timeSig, fmt='ro')
        plt.xlabel("Flowrate (SCCM)")
        plt.ylabel("Arrival time at z={} m".format(plane))
        plt.show()
        plt.clf()

        plt.title("Angular Spread vs Flowrate")
        plt.errorbar(x=frs, y=spreads, fmt='ro')
        plt.xlabel("Flowrate (SCCM)")
        plt.ylabel("Arrival time at z={} m".format(plane))
        plt.show()
        plt.clf()

        plt.title("Extraction Rate vs Gamma")
        plt.errorbar(x=gammas, y=ext, yerr=sigE,fmt='ro')
        plt.xlabel("Gamma")
        plt.ylabel("Fraction Extracted".format(plane))
        plt.show()
        plt.clf()

        plt.title("Forward Velocity vs Reynolds Number")
        plt.errorbar(x=reyn, y=vz, yerr=vzSig, fmt='ro')
        plt.xlabel("Reynolds Number")
        plt.ylabel("Forward Velocity (m/s)")
        plt.show()
        plt.clf()

        plt.title("Forward Velocity FWHM vs Reynolds Number")
        plt.errorbar(x=reyn, y=vzSig, fmt='ro')
        plt.xlabel("Reynolds Number")
        plt.ylabel("Forward Velocity St. Dev.")
        plt.show()
        plt.clf()

        plt.title("Forward Velocity FWHM vs Flowrate")
        plt.errorbar(x=frs, y=vzSig, fmt='ro')
        plt.xlabel("Flowrate (SCCM)")
        plt.ylabel("Forward Velocity St. Dev.")
        plt.show()
        plt.clf()

        plt.title("Reynolds Number vs Flowrate")
        plt.errorbar(x=frs, y=reyn, fmt='ro')
        plt.ylabel("Reynolds Number")
        plt.xlabel("Flowrate (SCCM)")
        plt.show()
        plt.clf()

        plt.title("Angular Spread vs Reynolds Number")
        plt.errorbar(x=reyn, y=spreads, fmt='ro')
        plt.xlabel("Reynolds Number")
        plt.ylabel("Calculated Spread")
        plt.show()
        plt.clf()

        plt.title("Theta Std. Dev. vs Reynolds Number")
        plt.errorbar(x=reyn, y=thetaSig, fmt='ro')
        plt.xlabel("Reynolds Number")
        plt.ylabel("Theta Stand. Dev.")
        plt.show()
        plt.clf()

        plt.title("OTHER Angular Spread vs Reynolds Number")
        plt.errorbar(x=reyn, y=spreadB, fmt='ro')
        plt.xlabel("Reynolds Number")
        plt.ylabel("Calculated Spread B")
        plt.show()
        plt.clf()


#import cProfile
#cProfile.run("endPosition(form='currentCell')")

# faster code?
# other dsmc programs?

# Parameters to vary:
#       Cross-sectional area to match experimental results
#           Find gammas at good cross, match fig 3 (patterson doyle)?
#       DSMC inputs (geometry & flow density)
#       Initial position & velocity of species (e.g. ablating - later)
#       Ambient velocity accuracy/precision (later)
