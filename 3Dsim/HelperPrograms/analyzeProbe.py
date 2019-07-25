from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

APERTURE_POS = 64.0 #z axis location of aperture in mm

def showPlot(dotSize, cellGeometry):

    lineWidth = 2.
    pos0 = 0.064
    folder = "C:/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/Probe/"
    # fileList = ['f5/Set1/probeResults.dat',\
    #             'f5/Set2/probeResults.dat',\
    #             'f5/Set3/probeResults.dat',\
    #             'f5/Set4/probeResults.dat']#,\
    #             # 'f5/probeAnalysis17.dat']


    fileList = ['h5/Set1/probeResults.dat',\
                'h5/Set2/probeResults.dat',\
                'h5/Set3/probeResults.dat',\
                'h5/Set4/probeResults.dat',\
                'h5/Set5/probeResults.dat',\
                'h5/Set6/probeResults.dat']


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
    print("Write to {} from {}".format(OUTFILE,READFILE))

    rf = np.loadtxt(directory + '\\' + READFILE, skiprows=1)

    try:
        outf = np.loadtxt(directory+'\\'+OUTFILE, skiprows=1)
        print("{0} particles recorded in {1}".format(len(outf), OUTFILE) )
    except StopIteration:
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
        with open(directory+'\\'+OUTFILE,'a') as tc:
            tc.write(''.join(map(str,list(results))))

    print("{0} made it, {1} did not".format(numYes, numNo))


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
    parser.add_argument('--plot', dest='plot', action='store_true', default=False)
    parser.add_argument('--dotsize', dest='dotsize', type=int, default=1)
    parser.add_argument('--geom', dest='geom', default='f')
    args=parser.parse_args()

    directory = args.direc
    OUTFILE = args.outfile
    READFILE = args.readfile
    write = args.write
    plot = args.plot
    dotSize = args.dotsize
    geometry = args.geom

    if plot:
        showPlot(dotSize, geometry)
    else:
        analyzeFiles(directory, OUTFILE, READFILE, write)
