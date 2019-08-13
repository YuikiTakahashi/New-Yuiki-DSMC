#! /usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import gaussian_kde
import argparse
#matplotlib.use("Agg")
#plt.rcParams['animation.ffmpeg_path'] = 'usr/bin/ffmpeg'
#FFwriter = animation.FFMpegWriter()
directory = "C:\\Users\\gabri\\Box\\HutzlerLab\\Data\\Woolls_BG_Sims\\Animation\\"
infile = 'mT_f100_75fr_th.dat'
outfile = 'f100_movie.mp4'

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'),bitrate=1800)
colorMap = 'plasma'
lineWidth = 1.0

interval_input = 250 #sets speed (frame rate?) of animation
nbins = 100

DEBUG = 0
SAVE=1  #should we save the animation
PLAY =1

#f is an Nx3 matrix where the columns are z, r, t
f = np.loadtxt(directory+infile, skiprows=1)

#Get the times for each of the "frames" recorded
times = np.unique(f[:,2])
numFrames = np.size(times)

molPositions = {}
for t in times:
    molPositions.update( {t : f[f[:,2] == t]} )



def animate(i):
    t_i = times[i]
    positions = molPositions[t_i]
    x = positions[:,1] #z coordinates
    y = positions[:,0] #r coordinates

    # counts, xedges, yedges, d = ax2.hist2d(x,y,range=range, bins=100)
    npdata, xedges, yedges = np.histogram2d(x,y,range=range,bins=100)
    # plt.clf()
    # im = ax.imshow(counts.T, origin='lower',extent=[0,120,0,30], aspect='auto', interpolation='gaussian', cmap='Blues')
    im = ax.imshow(npdata.T, origin='lower',extent=[0,120,0,30], aspect='auto', interpolation='gaussian', cmap=colorMap)
    # im.set_data(counts.T)
    #
    # data = np.vstack([Xi,Yi])
    # kde = gaussian_kde(data)
    # Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    # im = plt.imshow(Z.reshape(Xgrid.shape), origin='lower', aspect='auto', extent=[0, 120, 0, 30], cmap='Blues')

    # npdata,npx,npy = np.histogram2d(Xi, Yi, bins=nbins)
    # extent = [npx[0],npx[-1],npy[0],npy[-1]]
    # im = plt.imshow(Z.reshape(Xgrid.shape), origin='lower', aspect='auto', extent=[0, 120, 0, 30], cmap='Blues')
    # im.set_data(Z.reshape(Xgrid.shape))
    # im = plt.imshow(npdata, origin='lower', aspect='auto', extent=extent, cmap='Blues' )
    # dtxt.set_text('Frame {}'.format(i))

    # hist = plt.hist2d(x=positions[:,1], y=positions[:,0], bins=nbins, range=[[0,120],[0,30]])
    # return hist, dtxt

    return im



#

# animate(0)
# plt.show()

if DEBUG:
        t0 = times[70]
        positions = molPositions[t0]
        x = positions[:,1] #z coordinates
        y = positions[:,0] #r coordinates
        range=[[0,120],[0,30]]

        fig, (ax, ax2, ax3,ax4) = plt.subplots(ncols=4)


        for a in [ax,ax2,ax3,ax4]:
            a.vlines(1, 0, 1.5875, colors='gray', linewidths=.5)
            a.hlines(1.5875, 1, 15, colors='gray', linewidths=.5)
            a.vlines(15, 1.5875, 6.35, colors='gray', linewidths=.5)
            a.hlines(6.35, 15, 63.5, colors='gray', linewidths=.5)
            a.vlines(63.5, 6.35, 2.5, colors='gray', linewidths=.5)
            a.hlines(2.5, 63.5, 64, colors='gray', linewidths=.5)
            a.vlines(64, 2.5, 9, colors='gray', linewidths=.5)
            a.hlines(9, 0, 64, colors='gray', linewidths=.5)


        counts, xedges, yedges, d = ax.hist2d(x,y,range=range, bins=100)

        X,Y = np.meshgrid(xedges,yedges)
        mesh1 = ax2.pcolormesh(X,Y,counts.T)

        ybins, xbins = np.linspace(0,120,50), np.linspace(0,30,50)
        map, xedges, yedges = np.histogram2d(x,y,bins=[xbins,ybins])
        ax3.imshow(map, origin='lower',extent=[0,120,0,30], aspect='auto', interpolation='bilinear')
        ax4.imshow(counts.T, origin='lower',extent=[0,120,0,30], aspect='auto', interpolation='bilinear')

        plt.show()

if PLAY:

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
    fig, ax = plt.subplots()

    ax.vlines(1, 0, 1.5875, colors='gray', linewidths=lineWidth)
    ax.hlines(1.5875, 1, 15, colors='gray', linewidths=lineWidth)
    ax.vlines(15, 1.5875, 6.35, colors='gray', linewidths=lineWidth)
    ax.hlines(6.35, 15, 63.5, colors='gray', linewidths=lineWidth)
    ax.vlines(63.5, 6.35, 2.5, colors='gray', linewidths=lineWidth)
    ax.hlines(2.5, 63.5, 64, colors='gray', linewidths=lineWidth)
    ax.vlines(64, 2.5, 9, colors='gray', linewidths=lineWidth)
    ax.hlines(9, 0, 64, colors='gray', linewidths=lineWidth)
    im = ax.imshow(counts.T, origin='lower',extent=[0,120,0,30], aspect='auto', interpolation='gaussian', cmap=colorMap)

    anim = animation.FuncAnimation(fig, animate, frames=np.size(times), interval=interval_input, blit=False)

    if SAVE:
        anim.save(directory+outfile,writer=writer)


    plt.show()
