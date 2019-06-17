
import numpy as np
import scipy.stats as st
import scipy.interpolate as si
from joblib import Parallel, delayed
from multiprocessing import Pool
import itertools
import argparse
import matplotlib.pyplot as plt


def plot_dens(z0=0, zf = 70.0, size = 100,print_arrays=False):

    z_array = np.linspace(z0, zf, num=size)
    dens_array = np.ones(size)
    for i in range(size):
        dens_array[i] = get_dens(0, 0, z_array[i], f1)

    if print_arrays==True:
        print("Z array:")
        print(z_array)
        print("Density array:")
        print(dens_array)


    fig, ax = plt.subplots()
    ax.plot(z_array, dens_array)
    plt.show()
    

def main():

    print("Started main")

    parser = argparse.ArgumentParser('Simulation Specs')
    parser.add_argument('-ff', '--one') # Specify flowfield
    args = parser.parse_args()
    FF = args.one

    f1 = set_field(FF)
    plot_dens()
    


def get_dens(x0, y0, z0, func):
    quant0 = func(z0, (x0**2 + y0**2)**0.5)[0][0]
    return np.exp(quant0)


def set_field(FF):
    try:
        flowField = np.loadtxt(FF, skiprows=1) # Assumes only first row isn't data.
        print("Loaded flow field")

        zs, rs, dens, temps = flowField[:, 0], flowField[:, 1], flowField[:, 2], flowField[:, 7]

        print("1")
        vzs, vrs, vps = flowField[:, 4], flowField[:, 5], flowField[:, 6]
        quantHolder = [zs, rs, dens, temps, vzs, vrs, vps]
        print("2")
        grid_x, grid_y = np.mgrid[0.010:0.12:4500j, 0:0.030:1500j] # high density, to be safe.
        grid_dens = si.griddata(np.transpose([zs, rs]), np.log(dens), (grid_x, grid_y), 'nearest')
        print("3")
        #grid_temps = si.griddata(np.transpose([zs, rs]), temps, (grid_x, grid_y), 'nearest')
        #grid_vzs = si.griddata(np.transpose([zs, rs]), vzs, (grid_x, grid_y), 'nearest')
        #grid_vrs = si.griddata(np.transpose([zs, rs]), vrs, (grid_x, grid_y), 'nearest')
        #grid_vps = si.griddata(np.transpose([zs, rs]), vps, (grid_x, grid_y), 'nearest')

        #Interpolation functions:
        f1 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_dens)
        #f2 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_temps)
        #f3 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vzs)
        #f4 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vrs)
        #f5 = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vps)
        print("Thru block 2")
        return f1
    except:
        print("Note: No Flow Field DSMC data.")

#if __name__ == '__main__':
#    main()

