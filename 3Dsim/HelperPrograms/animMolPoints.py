#! /usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
#matplotlib.use("Agg")
#plt.rcParams['animation.ffmpeg_path'] = 'usr/bin/ffmpeg'
#FFwriter = animation.FFMpegWriter()
directory = "C:\\Users\\gabri\\Desktop\\HutzlerSims\\Gas-Simulation\\3Dsim\\Data\\WoollsData\\"
#infile = 'moleculeTracking_f00210.dat'
infile = 'molTr_f100_150_75frames.dat'
outfile = 'f100_movie.mp4'


interval_input = 250 #sets speed (frame rate?) of animation

SAVE=0  #should we save the animation

#f is an Nx3 matrix where the columns are z, r, t
f = np.loadtxt(directory+infile, skiprows=1)

#Get the times for each of the "frames" recorded
times = np.unique(f[:,2])
numFrames = np.size(times)

molPositions = {}
for t in times:
    molPositions.update( {t : ([], [])} )


for i in range(len(f)):
    r, z, t = f[i]
    arr = molPositions[t]
    rs, zs = arr[0], arr[1]
    rs += [r]
    zs += [z]
    molPositions.update( {t : (rs, zs)} )


fig = plt.figure()
ax = plt.axes(xlim=(0,120), ylim=(0,30))
lineS, = plt.plot([],[],'ro')
dtxt = plt.text(15, 9, "", fontsize=12)

plt.vlines(1, 0, 1.5875, colors='gray', linewidths=.5)
plt.hlines(1.5875, 1, 15, colors='gray', linewidths=.5)
plt.vlines(15, 1.5875, 6.35, colors='gray', linewidths=.5)
plt.hlines(6.35, 15, 63.5, colors='gray', linewidths=.5)
plt.vlines(63.5, 6.35, 2.5, colors='gray', linewidths=.5)
plt.hlines(2.5, 63.5, 64, colors='gray', linewidths=.5)
plt.vlines(64, 2.5, 9, colors='gray', linewidths=.5)
plt.hlines(9, 0, 64, colors='gray', linewidths=.5)


def init():
    lineS.set_data([],[])
    # dtxt.set_text("yeet")
    return lineS, dtxt,

def animate(i):
    t_i = times[i]
    positions = molPositions[t_i]
    lineS.set_data(positions[1], positions[0])
    # dtxt.set_text(cdata[i][2])
    return lineS, dtxt,


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=np.size(times), interval=interval_input, blit=True)

if SAVE:
    anim.save(savefile)

plt.show()
