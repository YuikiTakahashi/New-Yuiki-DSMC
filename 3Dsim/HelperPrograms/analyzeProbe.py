'''
Created summer 2019.

Code for compiling and plotting "probe" style: i.e. showing probability of
successful molecule extraction as a function of initial location in the cell.
'''


from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

APERTURE_POS = 64.0 #z axis location of aperture in mm
cellGeometry = 'h'

fileList = ['h5/Set1/probeResults.dat',\
            'h5/Set2/probeResults.dat',\
            'h5/Set3/probeResults.dat',\
            'h5/Set4/probeResults.dat',\
            'h5/Set5/probeResults.dat',\
            'h5/Set6/probeResults.dat',\
            'h5/Set7/probeResults.dat']

# fileList = ['f5/Set1/probeResults.dat',\
#             'f5/Set2/probeResults.dat',\
#             'f5/Set3/probeResults.dat',\
#             'f5/Set4/probeResults.dat']#,\
#             # 'f5/probeAnalysis17.dat']


def showDensityPlot(cellNum):
    '''
    Reads data from every file in fileList and plots it density style, i.e. averaging
    success rate over bins (number of bins determined by cellNum).
    '''
    lineWidth = 3.
    pos0 = 0.064
    folder = "C:/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/Probe/"

    arrayDic = {}
    totalParticles = 0
    finals = np.ones((0,4))

    for f in fileList:
        arr = np.loadtxt(folder + f, skiprows=1)
        arrayDic.update( {f : arr} )
        totalParticles += len(arr)
        finals = np.append(finals, arr, axis=0)
        print('{} has {} particles, finals now has {}'.format(f, len(arr), len(finals)))


    # rs = np.sqrt(f[:,0]**2 + f[:,1]**2)
    # zs, Ms = f[:,2], f[:,3]
    #
    # zMin, zMax, rMin, rMax = 0.0, 64.0, 0, 6.35
    # z_axis = np.linspace(zMin, zMax, num=cellNum, endpoint=True)
    # r_axis = np.linspace(rMin, rMax, num=cellNum, endpoint=True)

    success = finals[finals[:,3] == 1]
    failed = finals[finals[:,3] == 0]

    x1 = success[:, 0]
    y1 = success[:, 1]
    z1 = success[:, 2]
    r1 = np.sqrt(x1**2+y1**2)
    counts1, xedge1, yedge1 = np.histogram2d(z1, r1, range=[[0,64],[0,6.35]], bins=cellNum)

    x0 = failed[:, 0]
    y0 = failed[:, 1]
    z0 = failed[:, 2]
    r0 = np.sqrt(x0**2+y0**2)
    counts0, xedge0, yedge0 = np.histogram2d(z0, r0, range=[[0,64],[0,6.35]], bins=cellNum)

    fig, ax3 = plt.subplots()
    # im1 = ax1.imshow(counts0.T,origin='lower',extent=[0,64,0,6.35], aspect='auto',cmap='Reds')
    # im2 = ax2.imshow(counts1.T,origin='lower',extent=[0,64,0,6.35], aspect='auto',cmap='Greens')

    fullData = np.ones(counts1.shape)
    for i in range(fullData.shape[0]):
        for j in range(fullData.shape[1]):
            succs = counts1[i][j]
            fails = counts0[i][j]
            fullData[i][j] = succs/(succs+fails)

    im3 = ax3.imshow(fullData.T, origin='lower', extent=[0,64,0,6.35], aspect='auto', cmap='RdYlGn',interpolation='none')

    plt.vlines(1, 0, 1.5875, colors='gray', linewidths=lineWidth)
    plt.hlines(1.5875, 1, 15, colors='gray', linewidths=lineWidth)
    plt.vlines(15, 1.5875, 6.35, colors='gray', linewidths=lineWidth)
    if cellGeometry == 'f':
        ax3.hlines(6.35, 15, 63.5, colors='gray', linewidths=lineWidth)
        ax3.vlines(63.5, 6.35, 2.5, colors='gray', linewidths=lineWidth)
        ax3.hlines(2.5, 63.5, 64, colors='gray', linewidths=lineWidth)
        ax3.vlines(64, 2.5, 9, colors='gray', linewidths=lineWidth)
        ax3.hlines(9, 0, 64, colors='gray', linewidths=lineWidth)

    elif cellGeometry == 'h':
        ax3.hlines(6.35, 15, 59.65, colors='gray', linewidths=lineWidth)
        ax3.plot([59.65,63.5],[6.35,2.5], color='gray', linewidth=lineWidth)
        ax3.hlines(2.5, 63.5, 64, colors='gray', linewidths=lineWidth)
        ax3.plot([64,67.85],[2.5,6.35], color='gray', linewidth=lineWidth)
        ax3.vlines(67.85,6.35,9, colors='gray', linewidths=lineWidth)
        ax3.hlines(9, 0, 67.85, colors='gray', linewidths=lineWidth)


    ax3.set_xlim(0, 72)
    ax3.set_ylim(0, 10)
    plt.show()

def showWallPlot():
    lineWidth = 3.




def showPointPlot(dotSize):
    '''
    Reads data from every file in fileList and plots invidividual points, i.e.
    a green dot for a successful extraction, red for failed.
    '''

    lineWidth = 2.
    pos0 = 0.064
    folder = "C:/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/Probe/"

    arrayDic = {}
    totalParticles = 0
    finals = np.ones((0,4))

    for f in fileList:
        arr = np.loadtxt(folder + f, skiprows=1)
        arrayDic.update( {f : arr} )
        totalParticles += len(arr)
        finals = np.append(finals, arr, axis=0)
        print('{} has {} particles, finals now has {}'.format(f, len(arr), len(finals)))


    success = finals[finals[:,3] == 1]
    failed = finals[finals[:,3] == 0]

    x1 = success[:, 0] / 1000.
    y1 = success[:, 1] / 1000.
    z1 = success[:, 2] / 1000.

    x0 = failed[:, 0] / 1000.
    y0 = failed[:, 1] / 1000.
    z0 = failed[:, 2] / 1000.

    # xs = finals[:, 0] / 1000.
    # ys = finals[:, 1] / 1000.
    # zs = finals[:, 2] / 1000.
    green = plt.cm.Greens(175)
    red = plt.cm.Reds(120)
    plt.scatter(x=z1, y=np.sqrt(x1**2+y1**2), c=green, s=dotSize)
    plt.scatter(x=z0, y=np.sqrt(x0**2+y0**2), c=red, s=dotSize)

    plt.vlines(0.001, 0, 0.0015875, colors='gray', linewidths=lineWidth)
    plt.hlines(0.0015875, 0.001, 0.015, colors='gray', linewidths=lineWidth)
    plt.vlines(0.015, 0.0015875, 0.00635, colors='gray', linewidths=lineWidth)

    if cellGeometry == 'f':
        plt.hlines(0.00635, 0.015, 0.0635, colors='gray', linewidths=lineWidth)
        plt.vlines(0.0635, 0.00635, 0.0025, colors='gray', linewidths=lineWidth)
        plt.hlines(0.0025, 0.0635, 0.064, colors='gray', linewidths=lineWidth)
        plt.vlines(0.064, 0.0025, 0.009, colors='gray', linewidths=lineWidth)
        plt.hlines(0.009, 0, 0.064, colors='gray', linewidths=lineWidth)

    elif cellGeometry == 'h':
        plt.hlines(0.00635, 0.015, 0.05965, colors='gray', linewidths=lineWidth)
        plt.plot([0.05965,0.0635],[0.00635,0.0025], color='gray', linewidth=lineWidth)
        plt.hlines(0.0025, 0.0635, 0.064, colors='gray', linewidths=lineWidth)
        plt.plot([0.064,0.06785],[0.0025,0.00635], color='gray', linewidth=lineWidth)
        plt.vlines(0.06785,0.00635,0.009, colors='gray', linewidths=lineWidth)
        plt.hlines(0.009, 0, 0.06785, colors='gray', linewidths=lineWidth)

    plt.xlim(0, pos0+0.01)
    plt.ylim(0, 0.01)
    plt.show()
    plt.clf()

    return


def analyzeFiles(directory, OUTFILE, READFILE, write):
    '''
    Read file output of parSimArgs (run on PROBE_MODE and init_mode>=9) and write
    to a probeResults type file, with rows as [X Y Z M] (M = 0 or 1, indicating
    success or failure in extraction)
    '''

    print("Write to {} from {}".format(OUTFILE,READFILE))

    rf = np.loadtxt(directory + '\\' + READFILE, skiprows=1)

    try:
        outf = np.loadtxt(directory+'\\'+OUTFILE, skiprows=1)
        print("{0} particles recorded in {1}".format(len(outf), OUTFILE) )
    except:
        print("0 particles in {0}".format(OUTFILE))

    num_adding = count_particles(directory+'\\'+READFILE)
    print("Particles in {0}: {1}".format(READFILE, num_adding) )

    results = []

    x0, y0, z0 = rf[0][0], rf[0][1], rf[0][2]

    numYes, numNo = 0, 0

    for i in range(len(rf)):

        #When arrive at a row of zeros, save previous particle and reset
        if (not np.any(rf[i])):

            #File previous particle
            madeIt = rf[i-1][2] > APERTURE_POS

            if madeIt:
                results.append('  '.join(map(str, [round(x0,3), round(y0,3), round(z0,3), 1]) ) +'\n')
                numYes += 1

            else:
                results.append('  '.join(map(str, [round(x0,3), round(y0,3), round(z0,3), 0]) ) +'\n' )
                numNo += 1

            #Reset particle
            try:
                x0, y0, z0 = rf[i+1][0], rf[i+1][1], rf[i+1][2]
            except IndexError:
                print('')

    if write==True:
        with open(directory+'\\'+OUTFILE,'a+') as tc:
            tc.write(''.join(map(str,list(results))))

    print("{0} made it, {1} did not".format(numYes, numNo))


    #Count+print the number of particles/datapoints in the output file after the program has run
    try:
        outf = np.loadtxt(directory+'\\'+OUTFILE, skiprows=1)
        print("{0} particles recorded in {1}".format(len(outf), OUTFILE) )
    except StopIteration:
        print("0 particles in {0}".format(OUTFILE))





# ************************************************************************ #


if __name__ == '__main__':
    def count_particles(filename):
        f = np.loadtxt(filename, skiprows=1)
        num = 0 #number of simulated particles
        for i in range(len(f)):
            if not np.any(f[i]):
                num += 1 # count number of particles simulated in file
        return num


    parser = ArgumentParser()
    parser.add_argument('-folder', '--direc')
    parser.add_argument('-fout', '--outfile')
    parser.add_argument('-fin', '--readfile')
    parser.add_argument('--write', dest='write', action='store_true', default=False)
    parser.add_argument('--plot', dest='plotType', action='store', default=None)
    parser.add_argument('--dotsize', dest='dotsize', type=int, default=1)
    parser.add_argument('--cellnum', dest='cellNum', type=int, default=100)
    args=parser.parse_args()

    directory = args.direc
    OUTFILE = args.outfile
    READFILE = args.readfile
    write = args.write

    plot = args.plotType

    dotSize = args.dotsize
    cellNum = args.cellNum

    if plot == 'density':
        showDensityPlot(cellNum)
    elif plot == 'points':
        showPointPlot(dotSize)
    elif plot == 'wall':
        showWallPlot()
    elif plot == None:
        analyzeFiles(directory, OUTFILE, READFILE, write)
