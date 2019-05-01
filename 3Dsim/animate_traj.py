#! /usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
#matplotlib.use("Agg")
#plt.rcParams['animation.ffmpeg_path'] = 'usr/bin/ffmpeg'
#FFwriter = animation.FFMpegWriter()


parser = argparse.ArgumentParser('Simulation Specs') # parsimargs.py -ff "filename" -out "outfilename"
parser.add_argument('-in', '--one') # Specify flowfield
parser.add_argument('-out', '--two') # Specify output filename
parser.add_argument('-ivl', '--interval', nargs='?', const=10, default=10, type=int)
#parser.add_argument('--mult', type=int) # Specify cross section multiplier (optional)
args = parser.parse_args()

FF = args.one #file to read from
outfile = args.two #file to save to
interval_input = args.interval #sets speed (frame rate?) of animation

if args.one:
    GAVE_INPUT=1
    fileobj=open(FF) #file object to read data from
else:
    GAVE_INPUT=0
    fileobj=open('NewData/try_2.dat') #default file to read from, if an argument is not given

# if args.two:
#     SAVE=1
#     savefile=outfile
# else:
#     SAVE=0

SAVE=0  #should we save the animation

x_data = []
y_data = []

#fileobj = open('NewData/try_1.dat')
next(fileobj)

for line in fileobj:
    row = line.split()
    x_data.append(float(row[0]))
    y_data.append(float(row[1]))

x = np.array(x_data)
y = np.array(y_data)

xmin=np.amin(x)
ymin=np.amin(y)
xmax=np.amax(x)
ymax=np.amax(y)

#print("xmin: {}, xmax: {}, ymin: {}, ymax:{}".format(xmin,xmax,ymin,ymax))


fig = plt.figure()
ax = plt.axes(xlim=(xmin,xmax), ylim=(ymin,ymax))
lineS, = plt.plot([],[],'ro')
dtxt = plt.text(0, 9, "yeet", fontsize=12)

#x = range(2)

# xdata = np.loadtxt(xfilename)
# ydata = np.loadtxt(yfilename)
# cdata = np.loadtxt(cfilename)


#xdata = np.transpose(xdata)
#ydata = np.transpose(ydata)


def init():
    lineS.set_data([],[])
    dtxt.set_text("yeet")
    return lineS, dtxt,

def animate(i):
    lineS.set_data(x[i], y[i])
    # dtxt.set_text(cdata[i][2])
    return lineS, dtxt,


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x), interval=interval_input, blit=True)

if SAVE:
    anim.save(savefile)

plt.show()
