from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D






def plotWall(geometry):
    '''
    Plot final position data.
    '''
    APERTURE_POS = {'f':64, 'h':67.85, 'r':64}[geometry]
    NUM_BINS = 100

    f = np.loadtxt(directory+'\\'+readfile,skiprows=1)
    finals = np.zeros((0,2))

    for i in range(len(f)):
        if not np.any(f[i]):
            x, y, z = f[i-1,0], f[i-1,1], f[i-1,2]
            if z <= APERTURE_POS:
                r = np.sqrt(x**2+y**2)
                finals = np.append(finals, np.array([[r, z]]), axis=0)

    rs, zs = finals[:,0], finals[:,1]
    fig, (ax,ax2) = plt.subplots(ncols=2)
    ax.plot(zs, rs, '+', color='orange')
    showCellBoundaries(ax, geometry)

    data, xedges, yedges = np.histogram2d(zs, rs, range=[[0, APERTURE_POS+5.0],[0, 9]],bins=NUM_BINS)
    im2 = ax2.imshow(data.T, origin='lower',extent=[0, APERTURE_POS+5.0, 0, 9], aspect='auto',interpolation='gaussian',cmap='hot')
    showCellBoundaries(ax2, geometry)

    message = "Failed extractions. Flow r005, init = 1"
    plt.text(x=5,y=-1 ,s=message)
    plt.show()

def showCellBoundaries(ax, geometry):
    '''
    Plot lines to show outline of the cell
    '''

    lineWidth = 3.0

    if geometry == 'f':
        ax.vlines(1, 0, 1.5875, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.hlines(1.5875, 1, 15, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.vlines(15, 1.5875, 6.35, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.hlines(6.35, 15, 63.5, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.vlines(63.5, 6.35, 2.5, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.hlines(2.5, 63.5, 64, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.vlines(64, 2.5, 9, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.hlines(9, 0, 64, colors='gray', linewidths=lineWidth, alpha=0.3)

    elif geometry == 'h':
        ax.vlines(1, 0, 1.5875, colors='gray', linewidths=lineWidth)
        ax.hlines(1.5875, 1, 15, colors='gray', linewidths=lineWidth)
        ax.vlines(15, 1.5875, 6.35, colors='gray', linewidths=lineWidth)
        ax.hlines(6.35, 15, 59.65, colors='gray', linewidths=lineWidth)
        ax.plot([59.65,63.5],[6.35,2.5], color='gray', linewidth=lineWidth)
        ax.hlines(2.5, 63.5, 64, colors='gray', linewidths=lineWidth)
        ax.plot([64,67.85],[2.5,6.35], color='gray', linewidth=lineWidth)
        ax.vlines(67.85,6.35,9, colors='gray', linewidths=lineWidth)
        ax.hlines(9, 0, 67.85, colors='gray', linewidths=lineWidth)

    elif geometry == 'r':
        ax.vlines(1, 0, 1.5875, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.hlines(1.5875, 1, 15, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.vlines(15, 1.5875, 6.35, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.hlines(6.35, 15, 63.5, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.vlines(63.5, 6.35, 2.5, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.hlines(2.5, 63.5, 64, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.vlines(64, 2.5, 9, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.hlines(9, 0, 64, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.vlines(60.5, 6.35, 9, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.vlines(61, 6.35, 9, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.vlines(62, 6.35, 9, colors='gray', linewidths=lineWidth, alpha=0.3)
        ax.vlines(62.5, 6.35, 9, colors='gray', linewidths=lineWidth, alpha=0.3)
    else:
        raise ValueError("Need to specify boundaries for geometry {}".format(geometry))

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-folder', '--direc', default='C:\\Users\\gabri\\Box\\HutzlerLab\\Data\\Woolls_BG_Sims\\InitLarge')
    parser.add_argument('-fin', '--readfile',default='f005_init1.dat')
    args = parser.parse_args()
    directory = args.direc
    readfile = args.readfile

    geometry = readfile[0]

    plotWall(geometry)
