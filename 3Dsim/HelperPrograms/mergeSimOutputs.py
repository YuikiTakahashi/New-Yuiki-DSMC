from argparse import ArgumentParser
import numpy as np
import fileinput

def count_particles(filename):
    try:
        f = np.loadtxt(filename, skiprows=1)
        num = 0 #number of simulated particles
        for i in range(len(f)):
            if not np.any(f[i]):
                num += 1 # count number of particles simulated in file
    except:
        num=0
    return num


parser = ArgumentParser()
parser.add_argument('-folder', '--directory')
parser.add_argument('-fout', '--outfile')
parser.add_argument('-fin', '--file_list', nargs='+', default=[])
parser.add_argument('--app', dest='append', action='store_true', default=False)
args=parser.parse_args()

DIRECTORY = args.directory
OUTFILE = args.outfile
INFILE_LIST = args.file_list
APPEND = args.append

OUTFILE_FULL = DIRECTORY+'\\'+OUTFILE

INFILE_LIST_FULL = []
for inf in INFILE_LIST:
    INFILE_LIST_FULL.append(DIRECTORY+'\\'+inf)

print("Write to {} from {}, in directory {}".format(OUTFILE,INFILE_LIST,DIRECTORY))
print("Append is {}".format(APPEND))

try:
    print("Particles in {0}: {1}".format(OUTFILE, count_particles(OUTFILE_FULL)))
except StopIteration:
    print("0 particles in {0}".format(OUTFILE))

for f in INFILE_LIST:
    print("Particles in {0}: {1}".format(f, count_particles(DIRECTORY+'\\'+f)))

if APPEND==False:
    with open(OUTFILE_FULL, 'w+') as fout, fileinput.input(INFILE_LIST_FULL) as fin:
        HEADER = False
        for line in fin:
            if HEADER==False and line[0]=='x':
                fout.write(line)
                HEADER=True

            elif HEADER==True and line[0]!='x':
                fout.write(line)

            else:
                print("Skip")

elif APPEND==True:
    with open(OUTFILE_FULL, 'a') as fout, fileinput.input(INFILE_LIST_FULL) as fin:
        for line in fin:
            if line[0]!='x':
                fout.write(line)
            else:
                print('Skip')


print("Particles in {0}: {1}".format(OUTFILE, count_particles(OUTFILE_FULL)))
