from argparse import ArgumentParser
import numpy as np
import fileinput

APERTURE_POS = 64.0 #z axis location of aperture in mm

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
args=parser.parse_args()

directory = args.direc
OUTFILE = args.outfile
READFILE = args.readfile
write = args.write

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

    #When arrive at a row of zeros, file previous particle and reset
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
