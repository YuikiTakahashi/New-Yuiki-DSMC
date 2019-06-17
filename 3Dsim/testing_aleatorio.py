import argparse

if __name__ == '__main__':

    print("Started main")

    parser = argparse.ArgumentParser('Simulation Specs')
    parser.add_argument('-ff', '--one') # Specify flowfield
    parser.add_argument('-out', '--two') # Specify output filename

    parser.add_argument('--mult', type=int, dest='mult', action='store') # Specify cross section multiplier (optional)
    parser.add_argument('--npar', type=int, dest='npar', action='store') #Specify number of particles to simulate (optional, defaults to 1)
    parser.add_argument('--lite', dest='lite', action='store_true')
    parser.set_defaults(lite=False, mult=5, npar=1)
    args = parser.parse_args()

    FF = args.one
    outfile = args.two

    PARTICLE_NUMBER = args.npar
    crossMult = args.mult
    LITE_MODE = args.lite

    print("Step 1")
    print("Particle number {0}, crossmult {1}, LITE_MODE is {2}".format(PARTICLE_NUMBER,crossMult,LITE_MODE))

    # f = open(outfile, "w")
    # f.write('x (mm)   y (mm)   z (mm)   vx (m/s)   vy (m/s)   vz (m/s)   time (ms)   dens\n')
    # f.close()
