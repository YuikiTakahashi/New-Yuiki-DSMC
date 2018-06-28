#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:43:30 2018

@author: Dave
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as st

kb = 1.38 * 10**-23
NA = 6.022 * 10**23
T = 4 # Ambient temperature (K)
m = 0.004 / NA # Ambient gas mass (kg)
M = .190061 / NA # Species mass (kg)
massParam = 2 * m / (m + M)
n = 10**21 # m^-3
cross = 4 * np.pi * (140 * 10**(-12))**2 # two-helium cross sectional area
cross *= 4 # Rough estimate of He-YbOH cross sectional area

U = 1.5 * kb * T
vMean = 2 * (2 * kb * T / (m * np.pi))**0.5
vMeanM = 2 * (2 * kb * T / (M * np.pi))**0.5

coll_freq = n * cross * vMean
dt = 0.01 / coll_freq # ∆t satisfying E[# collisions in 100∆t] = 1.

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

# Maxwell-Boltzmann Velocity Distribution for ambient molecules
class vel_pdf(st.rv_continuous):
    def _pdf(self,x):
        return (m/(2*np.pi*kb*T))**1.5 * 4*np.pi * x**2 * np.exp(-m*x**2/(2*kb*T))  # x is velocity
vel_cv = vel_pdf(a=0, b=10**4, name='vel_pdf') # vel_cv.rvs() for value

# Set coordinate-dependent ambient flow average velocities
def ambientFlow(x, y, z, zFlow=0, xFlow=0, yFlow=0, aperture=False):
    if aperture == True:
        r = (x**2+y**2)**0.5
        radFlow = -5*z*np.exp(-0.4*abs(5*z+1)) * 100 * r
        xFlow = x * radFlow / r
        yFlow = y * radFlow / r
        zFlow = 0.2
    return (xFlow, yFlow, zFlow)

def collide():
    global vx, vy, vz, x, y, z
    dvx, dvy, dvz = ambientFlow(x, y, z)
    v = vel_cv.rvs() # Maxwell Distribution for ambient molecule
    Theta = Theta_cv.rvs()
    Phi = np.random.uniform(0, 2*np.pi)
    theta = theta_cv.rvs()
    phi = np.random.uniform(0, 2*np.pi)
    vx_amb = v * np.sin(theta) * np.cos(phi) + dvx - vx
    vy_amb = v * np.sin(theta) * np.sin(phi) + dvy - vy
    vz_amb = v * np.cos(theta) + dvz - vz
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

def getData(t1=500, t2=20000, step=500, trials=400, x0=0, y0=0, z0=0,\
            vx0=0, vy0=0, vz0=0):
    global vx, vy, vz, x, y, z
    '''
    For a variety of total time steps, determine the expected values of final position,
    square-distance, and path-averaged speed.
    '''
    with open('_'.join(map(str,[t1, t2, step, int(vx0), int(vy0), int(vz0), int(ambientFlow(0, 0, 0)[0]), \
                                int(M/m), int(np.log10(float(n)))]))+'.dat', 'w') as f:
        f.write('   '.join(['time (s)','xAvg (m)','yAvg (m)', 'zAvg (m)', 'SqrAvg (sq. m)',\
                          'SpeedAvg (m/s)','sigX','sigY', 'sigZ', 'sigSqr','sigSpeed'])+'\n')
        for time in range(t1, t2, step):
            print(time)
            xs = []
            ys = []
            zs = []
            squares = []
            speedAvgs = []
            for j in range(trials):
                x, y, z, vx, vy, vz = x0, y0, z0, vx0, vy0, vz0
                speeds = []
                count = 0
                while count < time:
                    if np.random.uniform() < 0.01: # 1/100 chance of collision
                        collide()
                    x += vx * dt
                    y += vy * dt
                    z += vz * dt
                    count += 1
                    if count > 0.8 * time:
                        speeds.append((vx**2+vy**2 + vz**2)**0.5)
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
            f.write(' '.join([str(time*dt),meanx,meany,meanz,meanSq,meanSpeed,stdx,stdy,stdz,stdSq,stdSpeed])+'\n')
    f.close()

def pathTrace(x0=0, y0=0, z0=0, vx0=0, vy0=0, vz0=0):
    global vx, vy, vz, x, y, z
    x, y, z, vx, vy, vz = x0, y0, z0, vx0, vy0, vz0
    xs = [0]
    ys = [0]
    zs = [0]

    # Assume a 10cm x 10cm x 10cm box
    in_box = True
    while in_box == True:
        # Typically takes few ms to leave box?
        if np.random.uniform() < 0.01: # 1/100 chance of collision
            collide()

        x += vx * dt
        y += vy * dt
        z += vz * dt
        xs.append(x)
        ys.append(y)
        zs.append(z)

        if abs(x) >= 0.05 or abs(y) >= 0.05 or abs(z) >= 0.05:
            in_box = False
    print("Iterations to wall: "+str(len(xs)))
    return xs, ys, zs

def getDensityTrend(filename, vel=True):
    global n, coll_freq, dt, vx, vy, vz, x, y, z
    with open(filename, 'w') as f:
        f.write("n (per cu. m)   mean iterations   mean time to stick   sig iter   sig time\n")
        for n in [10**9, 10**9*5, 10**10, 10**10*5, 10**11, 10**11*5, 10**12]:
            coll_freq = n * cross * vMean
            dt = 0.01 / coll_freq # ∆t satisfying E[# collisions in 100∆t] = 1.
            lens = []
            for j in range(1000):
                if vel == True:
                    xs, ys, zs = pathTrace(vx0=vMeanM)
                else:
                    xs, ys, zs = pathTrace()
                lens.append(len(xs))
            meanLen = str(np.mean(lens))
            meanTime = str(np.mean(lens)*dt)
            stdLen = str(np.std(lens)/(len(lens)**0.5))
            stdTime = str(np.std(lens)*dt/(len(lens)**0.5))
            f.write('%.1E'%n+' '+meanLen+' '+meanTime+' '+stdLen+' '+stdTime+'\n')
    f.close()

def getImage(filename, vel=True):
    global vx, vy, vz, x, y, z
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for j in range(1, 4):
        if vel == True:
            xs, ys, zs = pathTrace(vx0=vMeanM)
        else:
            xs, ys, zs = pathTrace()
        for i in range(0, len(xs), int(len(xs)/1000)):
            if j == 1 :
                colour = plt.cm.Greens(int(264. * i / len(xs)))
            elif j == 2:
                colour = plt.cm.Blues(int(264. * i / len(xs)))
            else:
                colour = plt.cm.Reds(int(264. * i / len(xs)))
            ax.scatter(xs[i], ys[i], zs[i], s=.5, c=colour)

    ax.set_xlim(-.05, .05)
    ax.set_ylim(-.05, .05)
    ax.set_zlim(-.05, .05)
    plt.title("Paths of a Light Particle, n = %.0E" %n)
    ax.set_xlabel('x, meters')
    ax.set_ylabel('y, meters')
    ax.set_zlabel('z, meters')
    plt.savefig(filename)
    plt.show()




#getData(vx0=vMeanM)


# Examine diffusion/pumpout ratio
# Relative velocity including VMeanM
# Increased collision probability at high velocity (Bayes?)
# Nonconstant flow
# Theory of x(t) distribution by matrix exponentiation?
