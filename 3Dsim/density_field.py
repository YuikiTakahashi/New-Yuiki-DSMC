import numpy as np
import scipy.interpolate as si
import argparse
import matplotlib.pyplot as plt


def main():
    global fdens, fmfp, Z_INFINITE, X0, Y0, SIZE, SIGMA
    print("Started main")

    parser = argparse.ArgumentParser('Simulation Specs')
    parser.add_argument('-ff', dest='ff', action='store') # Specify flowfield
    parser.set_defaults(ff='DS2FF017d.DAT')
    args = parser.parse_args()
    FF = args.ff

    fdens = {}
    fmfp = {}

    set_params(FF)
    plot_dens()

def set_params(FF='DS2FF017d.DAT', x=0, y=0):
    global fdens, fmfp, Z_INFINITE, X0, Y0, SIZE, SIGMA, flowrate
    X0 = x  #x, y coordinates in simulation site
    Y0 = y
    SIZE = 1000   #Size of z-arrays
    Z_INFINITE = 0.120  #Endpoint for integration, i.e. the "infinity" point
    SIGMA = 1e-14   #Cross section value

    flowrate = {'DS2FF017d.DAT':5, 'DS2FF018.DAT':20, 'DS2FF019.DAT':50, 'DS2FF020.DAT':10,\
                'DS2FF021.DAT':2, 'DS2FF022.DAT':100, 'DS2FF023.DAT':200}[FF]

    new_fdens, new_fmfp = set_field(FF)
    fdens.update({flowrate:new_fdens})
    # fdens[flowrate] = new_fdens

    fmfp.update({flowrate:new_fmfp})
    # fmfp[flowrate] = new_fmfp


##############################################################################
##********************** Flow field functions ***************************##
##############################################################################


def get_dens(x, y, z, which_flow=5):
    global fdens
    d_field = fdens[which_flow]
    quant0 = d_field(z, (x**2 + y**2)**0.5)[0][0]
    return np.exp(quant0)

def get_mfp(x, y, z, which_flow=5):
    global fmfp
    mfp_field = fmfp[which_flow]
    quant0 = mfp_field(z, (x**2 + y**2)**0.5)[0][0]
    return quant0

def set_field(FF):
    try:
        flowField = np.loadtxt('flows/'+FF, skiprows=1) # Assumes only first row isn't data.
        print("Loaded flow field")

        zs, rs, dens = flowField[:, 0], flowField[:, 1], flowField[:, 2]

#        vzs, vrs, vps = flowField[:, 4], flowField[:, 5], flowField[:, 6]
#        quantHolder = [zs, rs, dens, temps, vzs, vrs, vps]

        mfps = flowField[:,14]

        grid_x, grid_y = np.mgrid[0.010:0.12:4500j, 0:0.030:1500j] # high density, to be safe.
        grid_dens = si.griddata(np.transpose([zs, rs]), np.log(dens), (grid_x, grid_y), 'nearest')

        grid_mfp = si.griddata(np.transpose([zs, rs]), mfps, (grid_x, grid_y), 'nearest')

        #Interpolation functions:
        fdens = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_dens)
        fmfp = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_mfp)
        return fdens, fmfp

    except:
        print("Note: Failed Loading Flow Field DSMC data.")

##############################################################################
##********************** Plotting functions  *******************************##
##############################################################################

def plot_dens(x0=0, y0=0, z0=0, zf=0.15, array_size = 100, which_flow=5, print_arrays=False,log_scale=True):
    global fdens, flowrate
    z_array = np.linspace(z0, zf, num=array_size)
    dz = (zf-z0)/array_size
    print("Dz = {0}".format(dz))

    density = np.ones(array_size)
    for i in range(array_size):
        density[i] = get_dens(x0, y0, z_array[i], which_flow)

    if print_arrays:
        print("Z array:")
        print(z_array)
        print("Density array:")
        print(density)


    fig, ax = plt.subplots()
    ax.plot(z_array, density)
    plt.title("Buffer Gas Number Density Outside Cell \n Flowrate = {} SCCM".format(which_flow))
    if log_scale:
        plt.yscale('Log')
        plt.ylabel("Log Density (cm^-3)")
    else:
        plt.ylabel("Density (cm^-3)")

    plt.xlabel('Z distance (m)')
    plt.axvline(x=0.064)
    plt.show()

def plot_mfp(x0=0, y0=0, z0=0, zf=0.15, array_size = 100, which_flow=5, print_arrays=False, log_scale=True):
    global fmfp

    z_array = np.linspace(z0, zf, num=array_size)

    mfpath = np.ones(array_size)
    for i in range(array_size):
        mfpath[i] = get_mfp(x0, y0, z_array[i], which_flow)

    if print_arrays:
        print("Z array:")
        print(z_array)
        print("Mean free path array:")
        print(mfpath)


    fig, ax = plt.subplots()
    ax.plot(z_array, mfpath)
    plt.title("Mean Free Path in Buffer Gas \n Flowrate = {} SCCM".format(which_flow))

    if log_scale:
        plt.yscale('Log')
        plt.ylabel('Log Mean Free Path (cm?)')
    else:
        plt.ylabel('Mean Free Path (cm?)')

    plt.xlabel('Z distance (m)')
    plt.axvline(x=0.064)
    plt.show()
##########################################################


def get_ncoll(z0=0.064, zf=0.120, which_flow=5):
    global SIGMA

    z_array = np.linspace(z0, zf, num=SIZE)
    density = np.ones(SIZE)
    for i in range(SIZE):
        density[i] = get_dens(X0, Y0, z_array[i], which_flow)

    prob = SIGMA*density

    ncol = np.trapz(y=prob, x=z_array)
    return ncol

def plot_ncoll(numpoints=100, which_flow=5):

    plot_zrange = np.linspace(0.064, 0.120, num=numpoints)
    plot_ncol = np.ones(numpoints)
    for i in range(numpoints):
        current_z = plot_zrange[i]
        plot_ncol[i] = get_ncoll(z0=current_z, zf=0.120, which_flow=which_flow)
    fig, ax = plt.subplots()

    ax.plot(plot_zrange, plot_ncol)
    plt.yscale('Log')
    plt.ylabel('Log Collision Number')
    plt.xlabel('Z distance (m)')
    plt.show()





#if __name__ == '__main__':
#    main()
