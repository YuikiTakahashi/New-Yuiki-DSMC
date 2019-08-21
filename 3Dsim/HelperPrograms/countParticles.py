import numpy as np
import argparse

'''
Takes a particle trajectory data file (i.e. output from parSimArgs) and reports
number of simulated particles by counting the number of [0,0,0,0,0,0] rows.
'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Simulation Specs')
    parser.add_argument('-ff', '--one') # Specify flowfield
    args = parser.parse_args()
    FF = args.one

    f = np.loadtxt(FF, skiprows=1)
    num = 0 #number of simulated particles
    for i in range(len(f)):
        if not np.any(f[i]):
            num += 1 # count number of particles simulated in file
    print("Number of particles: {}".format(num))
