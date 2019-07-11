from argparse import ArgumentParser
import numpy as np
import fileinput

def count_particles(filename):
    f = np.loadtxt(filename, skiprows=1)
    num = 0 #number of simulated particles
    for i in range(len(f)):
        if not np.any(f[i]):
            num += 1 # count number of particles simulated in file
    return num


parser = ArgumentParser()
parser.add_argument('-fout', '--outfile')
parser.add_argument('-fin', '--file_list', nargs='+', default=[])
args=parser.parse_args()

OUTFILE = args.outfile
INFILE_LIST = args.file_list

print(OUTFILE)
print(INFILE_LIST)

for f in INFILE_LIST:
    print("Particles in {0}: {1}".format(f, count_particles(f)))

with open(OUTFILE, 'w') as fout, fileinput.input(INFILE_LIST) as fin:
    HEADER = False
    for line in fin:
        if HEADER==False and line[0]=='x':
            fout.write(line)
            HEADER=True

        elif HEADER==True and line[0]!='x':
            fout.write(line)

        else:
            print("Skip")

print("Particles in {0}: {1}".format(OUTFILE, count_particles(OUTFILE)))
