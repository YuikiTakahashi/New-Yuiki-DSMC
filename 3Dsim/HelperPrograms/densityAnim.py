#! /usr/bin/env python3

'''
Created summer 2019.

Takes file output from moleculeTracking.py, rows consisting of [Radius  Z  Time].
Reads and animates the file as a series of histograms showing the particle distribution
at each snapshot in time.
'''


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import gaussian_kde
import argparse

directory = "C:\\Users\\gabri\\Box\\HutzlerLab\\Data\\Woolls_BG_Sims\\Animation\\"
infile = 'mT_f005_75fr_th.dat'
outfile = 'f005_movie2.mp4'

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'),bitrate=1800)
colorMap = 'plasma'
lineWidth = 1.0

interval_input = 250 #sets speed (frame rate?) of animation
nbins = 100

GEOMETRY = 'f'
DEBUG = 0
SAVE=1  #should we save the animation
PLAY =1


def inBounds(x, y, geom='f'):

    if geom=='f':
        in1 = x > 15.0 and x < 63.5 and y < 6.35
        in2 = x > 63.5 and x < 64.0 and y < 2.5
        in3 = x > 64.0 and x < 120.0 and y < 30.0
        return in1+in2+in3

def animate(i):
    t_i = times[i]
    positions = molPositions[t_i]
    x = positions[:,1] #z coordinates
    y = positions[:,0] #r coordinates

    wallPos = wallPositions[t_i]
    xwall, ywall = wallPos[:,1], wallPos[:,0]

    # counts, xedges, yedges, d = ax2.hist2d(x,y,range=range, bins=100)
    npdata, xedges, yedges = np.histogram2d(x,y,range=range,bins=100)
    im = ax.imshow(npdata.T, origin='lower',extent=[0,120,0,30], aspect='auto', interpolation='gaussian', cmap=colorMap)
    pt = ax.plot(xwall, ywall, '.r', ms=2)

    npdata2, xedges2, yedges2 = np.histogram2d(xwall, ywall, range=range,bins=100)
    im2 = ax2.imshow(npdata2.T, origin='lower',extent=[0,120,0,30], aspect='auto', interpolation='gaussian')
    return im



if PLAY:

    #f is an Nx3 matrix where the columns are z, r, t
    f = np.loadtxt(directory+infile, skiprows=1)

    #Get the times for each of the "frames" recorded
    times = np.unique(f[:,2])
    numFrames = np.size(times)

    molPositions = {}
    for t in times:
        molPositions.update( {t : f[f[:,2] == t]} )

    # #want them to appear in one frame as wall dots but no more than one frame
    wallPositions = {}

    #for each frame i.e. snapshot
    for i in range(numFrames):
        t = times[i]
         #Get the list of molecule positions
        currentMolPos = molPositions[t]

        #Array to store the out-of-bounds (stuck to wall) particles
        currentWallPos = np.zeros((0,3))

         #For each particle i.e. row in the molecule positions array
        ptally, p = 0, 0
        total = len(currentMolPos)
        while ptally < total:
             #If the particle is out of bounds, remove it from the molPositions array
             #and add it to the wallPositions array
            x, y = currentMolPos[p,1], currentMolPos[p,0]

            if not inBounds(x, y, GEOMETRY):
                currentWallPos = np.append(currentWallPos, [[y, x, t]],axis=0)
                currentMolPos = np.delete(currentMolPos, obj=p, axis=0)

            else:
                p += 1 #move on to next row

            ptally += 1

        molPositions.update( {t : currentMolPos} )
        wallPositions.update( {t : currentWallPos} )
        #print(currentWallPos.size+currentMolPos.size)
        # print("Wall positions at t={}    :   ".format(t))
        # print(currentWallPos)


    xgrid = np.linspace(0,120,50)
    ygrid = np.linspace(0,30,50)
    Xgrid, Ygrid = np.meshgrid(xgrid,ygrid)

    t_i = times[0]
    positions = molPositions[t_i]
    Xi = positions[:,1] #z coordinates
    Yi = positions[:,0] #r coordinates
    range=[[0,120],[0,30]]

    counts, xedges, yedges = np.histogram2d(Xi,Yi,range=range, bins=100)
    # plt.clf()
    fig, (ax,ax2) = plt.subplots(ncols=2)

    ax.vlines(1, 0, 1.5875, colors='gray', linewidths=lineWidth)
    ax.hlines(1.5875, 1, 15, colors='gray', linewidths=lineWidth)
    ax.vlines(15, 1.5875, 6.35, colors='gray', linewidths=lineWidth)
    ax.hlines(6.35, 15, 63.5, colors='gray', linewidths=lineWidth)
    ax.vlines(63.5, 6.35, 2.5, colors='gray', linewidths=lineWidth)
    ax.hlines(2.5, 63.5, 64, colors='gray', linewidths=lineWidth)
    ax.vlines(64, 2.5, 9, colors='gray', linewidths=lineWidth)
    ax.hlines(9, 0, 64, colors='gray', linewidths=lineWidth)
    ax2.vlines(1, 0, 1.5875, colors='gray', linewidths=lineWidth)
    ax2.hlines(1.5875, 1, 15, colors='gray', linewidths=lineWidth)
    ax2.vlines(15, 1.5875, 6.35, colors='gray', linewidths=lineWidth)
    ax2.hlines(6.35, 15, 63.5, colors='gray', linewidths=lineWidth)
    ax2.vlines(63.5, 6.35, 2.5, colors='gray', linewidths=lineWidth)
    ax2.hlines(2.5, 63.5, 64, colors='gray', linewidths=lineWidth)
    ax2.vlines(64, 2.5, 9, colors='gray', linewidths=lineWidth)
    ax2.hlines(9, 0, 64, colors='gray', linewidths=lineWidth)
    im = ax.imshow(counts.T, origin='lower',extent=[0,120,0,30], aspect='auto', interpolation='gaussian', cmap=colorMap)

    anim = animation.FuncAnimation(fig, animate, frames=np.size(times), interval=interval_input, blit=False)

    if SAVE:
        anim.save(directory+outfile,writer=writer)


    plt.show()
