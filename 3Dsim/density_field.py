import numpy as np
import scipy.interpolate as si
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
from scipy.optimize import minimize_scalar
from functools import partial

# =============================================================================
# Always call main() first so that the quantity dictionaries get created
# =============================================================================

def main():
    global fdens, fmfp, ftemp, fvz, fvr, fvp, Z_INFINITE, X0, Y0, SIZE, SIGMA, get_quantity_dic
    print("Started main")

    parser = argparse.ArgumentParser('Simulation Specs')
    parser.add_argument('-ff', dest='ff', action='store') # Specify flowfield
    parser.set_defaults(ff='F_Cell/DS2f005.DAT')
    args = parser.parse_args()
    FF = args.ff

    fdens = {}
    fmfp = {}
    ftemp = {}
    fvz = {}
    fvr = {}
    fvp = {}

    set_params(FF)

    get_quantity_dic = {'dens':get_dens, 'temp':get_temp, 'vz':get_vz, 'mfp':get_mfp}

    plot_dens()

def set_many():
    flist = ['010','020','050','100','002','005','200']
    for f in flist:
        set_params(FF='H_Cell/DS2h{}.DAT'.format(f))

def set_params(FF='F_Cell/DS2f005.DAT', x=0, y=0):
    global fdens, fmfp, ftemp, fvz, fvr, fvp, Z_INFINITE, X0, Y0, SIZE, SIGMA
    X0 = x  #x, y coordinates in simulation site
    Y0 = y
    SIZE = 1000   #Size of z-arrays
    Z_INFINITE = 0.120  #Endpoint for integration, i.e. the "infinity" point
    SIGMA = 1e-14   #Cross section value

    #e.g. FF = F_Cell/DS2f020.DAT  or G_Cell/DS2g200.DAT
    flowtype = FF[-8:-4] #e.g. 'f020'

    #flowrate = {'DS2FF017d.DAT':5, 'DS2FF018.DAT':20, 'DS2FF019.DAT':50, 'DS2FF020.DAT':10,\
    #            'DS2FF021.DAT':2, 'DS2FF022.DAT':100, 'DS2FF023.DAT':200, 'DS2FF024.DAT':201}[FF]

    new_fdens, new_fmfp, new_ftemp, new_fvz, new_fvr, new_fvp = set_field(FF)

    fdens.update({flowtype:new_fdens})
    ftemp.update({flowtype:new_ftemp})
    fvz.update({flowtype:new_fvz})
    fvr.update({flowtype:new_fvr})
    fvp.update({flowtype:new_fvp})
    fmfp.update({flowtype:new_fmfp})


##############################################################################
##********************** Flow field functions ***************************##
##############################################################################

def set_field(FF):
    try:
        flowField = np.loadtxt('flows/'+FF, skiprows=1) # Assumes only first row isn't data.
        print("Loaded flow field {}".format(FF))

        zs, rs, dens, temps = flowField[:, 0], flowField[:, 1], flowField[:, 2], flowField[:,7]

        vzs, vrs, vps = flowField[:, 4], flowField[:, 5], flowField[:, 6]
#        quantHolder = [zs, rs, dens, temps, vzs, vrs, vps]

        mfps = flowField[:,14]

        print("Block 1: density and temperature")

        #Recall FF e.g.= F_Cell/DS2f200.DAT or G_Cell/DS2g005.DAT
        if FF[10] in ['f', 'g']:
            print('{} geometry grid'.format(FF[10]))
            grid_x, grid_y = np.mgrid[0.010:0.12:4500j, 0:0.030:1500j] # high density, to be safe.

        elif FF[10] in ['h', 'j', 'k']:
            print('{} geometry grid'.format(FF[10]))
            grid_x, grid_y = np.mgrid[0.010:0.24:9400j, 0:0.030:1500j] # high density, to be safe.


        grid_dens = si.griddata(np.transpose([zs, rs]), np.log(dens), (grid_x, grid_y), 'nearest')
        grid_temps = si.griddata(np.transpose([zs, rs]), temps, (grid_x, grid_y), 'nearest')

        print("Block 2: velocities and mfp")

        grid_vzs = si.griddata(np.transpose([zs, rs]), vzs, (grid_x, grid_y), 'nearest')
        grid_vrs = si.griddata(np.transpose([zs, rs]), vrs, (grid_x, grid_y), 'nearest')
        grid_vps = si.griddata(np.transpose([zs, rs]), vps, (grid_x, grid_y), 'nearest')

        grid_mfp = si.griddata(np.transpose([zs, rs]), mfps, (grid_x, grid_y), 'nearest')

        print("Interpolating")

        #Interpolation functions:
        fdens = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_dens)
        fmfp = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_mfp)
        ftemp = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_temps)
        fvz = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vzs)
        fvr = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vrs)
        fvp = si.RectBivariateSpline(grid_x[:, 0], grid_y[0], grid_vps)
        return fdens, fmfp, ftemp, fvz, fvr, fvp

    except:
        print("Note: Failed Loading Flow Field DSMC data.")



def get_dens(x, y, z, which_flow='f005'):
    global fdens
    d_field = fdens[which_flow]
    logdens = d_field(z, (x**2 + y**2)**0.5)[0][0]
    return np.exp(logdens)


def get_mfp(x, y, z, which_flow='f005'):
    global fmfp
    mfp_field = fmfp[which_flow]
    return mfp_field(z, (x**2 + y**2)**0.5)[0][0]


def get_temp(x, y, z, which_flow='f005'):
    global ftemp
    temp_field = ftemp[which_flow]
    return temp_field(z, (x**2+y**2)**0.5)[0][0]


def get_vz(x, y, z, which_flow):
    global fvz
    vz_field = fvz[which_flow]
    return vz_field(z, (x**2+y**2)**0.5)[0][0]

# =============================================================================
# Single-field plotting functions
# =============================================================================

def plot_dens(x0=0, y0=0, z0=0.010, zf=0.15, array_size = 100, which_flow='f005', print_arrays=False,log_scale=True):
    global fdens
    z_array = np.linspace(z0, zf, num=array_size)
#    dz = (zf-z0)/array_size
#    print("Dz = {0}".format(dz))

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



def plot_mfp(x0=0, y0=0, z0=0.010, zf=0.15, array_size = 100, which_flow='f005', print_arrays=False, log_scale=True):
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


def plot_quant_field(quantity="temp", rmax=0.01, z1=0.010, z2=0.15, array_size=50, which_flow='f005', logscale=False):

    if quantity in ["density", "dens"]:
        plot_density_field(rmax, z1, z2, array_size, which_flow)

    else:
        global fvz, ftemp, fmfp
        fquant = {"vz":fvz, "temp":ftemp, "mfp":fmfp}[quantity]

        quant_field = fquant[which_flow]

        z_axis = np.linspace(z1, z2, num=array_size)
        r_axis = np.linspace(0.0, rmax, num=array_size)

        zv, rv = np.meshgrid(z_axis, r_axis)
        quants = np.ones(zv.shape)

        for i in range(array_size):
            for j in range(array_size):
                z = zv[i,j]
                r = rv[i,j]
                quants[i,j] = quant_field(z, r)[0][0]

        if logscale:
            plt.pcolormesh(zv, rv, quants, norm=colors.LogNorm(vmin=quants.min(),vmax=quants.max() ) )
        else:
            plt.pcolormesh(zv, rv, quants)
        plt.show()

def plot_density_field(rmax=0.01, z1=0.010, z2=0.15, array_size=50, which_flow='f005', logscale=False):

    z_axis = np.linspace(z1, z2, num=array_size)
    r_axis = np.linspace(0.0, rmax, num=array_size)

    zv, rv = np.meshgrid(z_axis, r_axis)
    dens = np.ones(zv.shape)

    for i in range(array_size):
        for j in range(array_size):
            z = zv[i,j]
            r = rv[i,j]
            dens[i,j] = get_dens(0, r, z, which_flow)

    #print(dens)
    if logscale:
        plt.pcolormesh(zv, rv, dens, norm=colors.LogNorm(vmin=dens.min(),vmax=dens.max() ) )
    else:
        plt.pcolormesh(zv, rv, dens)

    plt.show()

# =============================================================================
# Get FWHM, or half-radius, in which half of the buffer gas is contained
# =============================================================================

def halfRadius(z, which_flow='h002', plot=False, pr=False):

    ARRAY_SIZE = 1000
    MAX_RAD = 0.03

    
    global fdens
    dens_field = fdens[which_flow]

    ds =  np.ones(ARRAY_SIZE)
    rs = np.sqrt( np.linspace(0, MAX_RAD**2, num=ARRAY_SIZE) )

    for i in range(ARRAY_SIZE):
        ds[i] = np.exp(dens_field(z, rs[i])[0][0])


    total_integral = integrate_cross_section(z, 0, MAX_RAD, which_flow)

    toMinPartial = partial(toMinimize, z=z, r1=0, which_flow=which_flow, targ=total_integral)
    res = minimize_scalar(toMinPartial, bounds=(0, 0.03), method='bounded')

    rHalf = res.x
    halfInt = integrate_cross_section(z, 0, rHalf, which_flow)

    if pr:
        print("Half-radius {}, integrates to {}".format(1000*rHalf, round(halfInt/total_integral,3)))

    if plot:
        plt.title('Radial Cross Section\n(z = {} mm)'.format(1000*z))
        plt.plot(1000*rs, ds)
        plt.xlabel('Radius [mm]')
        plt.ylabel(r'Number Density [$m^{-3}$]')
        plt.vlines(x=1000*rHalf, ymin=0, ymax=ds.max(), colors='r',label=r'$R_{1/2} =$'+' {} mm'.format(round(1000*rHalf,3)))
        plt.legend()
        plt.show()    
        
    return rHalf

def toMinimize(x, z, r1, which_flow, targ):
    return abs(0.5-(integrate_cross_section(z=z, r1=r1, r2=x, which_flow=which_flow) / targ))


def integrate_cross_section(z, r1, r2, which_flow):
    ARRAY_SIZE=1000
    global fdens
    dens_field = fdens[which_flow]

    rs = np.sqrt( np.linspace(r1**2, r2**2, num=ARRAY_SIZE) )
    ds =  np.ones(ARRAY_SIZE)

    for i in range(ARRAY_SIZE):
        ds[i] = np.exp(dens_field(z, rs[i])[0][0])

    return np.trapz(y=ds*rs, x=rs)

############################################################################################################
############################################################################################################
    

# =============================================================================
# Multi-field plotting functions, for comparing different geometries/flowrates
# at the aperture
# =============================================================================

def multi_plot_quant(quantity='dens', flowList=['f005','g200'], z0=0.010, zf=0.15, logscale=True):

    global get_quantity_dic
    getter = get_quantity_dic[quantity] #select method get_dens, get_mfp, etc

    array_size=200

    title = {'mfp':'Mean Free Path', 'vz':'Forward Velocity', 'temp':'Temperature','dens':'Density'}[quantity]

    fig, ax = plt.subplots()
    plt.title('Buffer Gas '+title+'\nde Laval aperture')

    z_array = np.linspace(z0,zf,num=array_size)
    #plt.title("Mean Free Path in Buffer Gas \n Flowrate = {} SCCM".format(which_flow))
#    legends = {'f200' : 'Straight hole, 200 SCCM',\
#               'g200' : 'Bevel hole, 200 SCCM', \
#               'f005' : 'Straight hole, 5 SCCM',\
#               'g005' : 'Bevel hole, 5 SCCM',\
#               'h200' : 'de Laval, 200 SCCM',\
#               'h005' : 'de Laval, 5 SCCM'}
    legends={}
    for flow in flowList:
        if flow not in legends:
#            flowtype = {'f': 'Straight hole, ',\
#                        'g': 'Bevel hole, ',\
#                        'h': 'de Laval, '}[flow[0]]
            flowtype=''
            flowrate = str(int(flow[1:4]))+' SCCM'
            legends.update( {flow : flowtype+flowrate})

    for f in flowList:

        quant_array = np.ones(array_size)
        for i in range(array_size):
            quant_array[i] = getter(x=0,y=0,z=z_array[i], which_flow=f)

        ax.plot(z_array, quant_array, label=legends[f])


    if logscale:
        plt.yscale('Log')
        plt.ylabel('Log '+title)
    else:
        plt.ylabel(title)

    plt.xlabel('Z distance (m)')
    plt.legend()
    #plt.axvline(x=0.064)
    plt.axvline(x=0.098)
    plt.show()


# =============================================================================
# Measure on window
# =============================================================================
def get_window_stats(file = 'Hcell.dat', z=0.094, which_flow='f005', write=0, plot=0):
    '''
    Returns statistics on vz, FWHM vz, angular spread, etc for a specified
    BG flow field, on a small window of radius WINDOW_RAD about the z-axis.
    
    plot = 1: Plots flow-dependent statistics for (file) geometry only.   
    plot = 3: Plots statistics for every geometry in (fileList)
    plot = 0: Plots velocities vs radius only for (which_flow)
       write = 1: Writes statistics of (which_flow) to a row in (file)
    '''
    WINDOW_RAD=0.03
    ARRAY_SIZE = 2000

    logscale=0

#    fileList = ['Fcell.dat', 'Gcell.dat', 'Hcell.dat']
    fileList = ['Fcell_plane.dat', 'Gcell_plane.dat', 'Hcell_plane.dat', 'Jcell_plane.dat']

    legends = {fileList[0] : 'Straight Hole',\
               fileList[1] : 'Beveled Aperture',\
               fileList[2] : 'de Laval I (H)',\
               fileList[3] : 'de Laval II (J)'}

    formats = {fileList[0] : 'go',\
               fileList[1] : 'ro',\
               fileList[2] : 'co',\
               fileList[3] : 'yo'}

    linestyles = {fileList[0] : '--',\
                  fileList[1] : '--',\
                  fileList[2] : ':',\
                  fileList[3] : ':'}


    fr_dic, ext_dic, sigE_dic, vr_dic, vz_dic, vzSig_dic, spreadB_dic, vrSig_dic,rhalf_dic = {},{},{},{},{},{},{},{},{}

    if plot==1:
        folder = '/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/BGWindow/'
        f = np.loadtxt(folder + file, skiprows=1)

        zs, frs, vz, vzSig, vR, vRSig, spreadB, rHalf = f[:,0], f[:,1], f[:,2], \
        f[:,3], f[:,4], f[:,5], f[:,6], f[:,7]

        plt.title("Angular Spread vs Flow")
        plt.errorbar(x=frs, y=spreadB, fmt='ro')
        plt.xlabel("Flow [SCCM]")
        plt.ylabel("Divergence [deg]")
        plt.show()
        plt.clf()

        plt.title("Forward Velocity vs Flow")
        plt.errorbar(x=frs, y=vz, yerr=vzSig, fmt='ro')
        plt.xlabel("Flow [SCCM]")
        plt.ylabel("Forward Velocity [m/s]")
        plt.show()
        plt.clf()

        plt.title("Forward Velocity FWHM vs Flow")
        plt.errorbar(x=frs, y=2.355*vzSig, fmt='ro')
        plt.xlabel("Flow [SCCM]")
        plt.ylabel("Forward Velocity FWHM [m/s]")
        plt.show()
        plt.clf()
        
        plt.title("Gas FWHM vs Flow\n(Plane 3cm past aperture)")
        plt.errorbar(x=frs, y=rHalf, fmt='ro')
        plt.xlabel("Flow [SCCM]")
        plt.ylabel("Half Radius [mm]")
        plt.show()
        plt.clf()



    elif plot==3:

        for file in fileList:
            folder = '/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/BGWindow/'
            f = np.loadtxt(folder + file, skiprows=1)

            zs, frs, vz, vzSig, vr, vRSig, spreadB, rHalf = f[:,0], f[:,1], f[:,2], \
            f[:,3], f[:,4], f[:,5], f[:,6], f[:,7]

            fr_dic.update( {file : frs} )

            vz_dic.update( {file : vz} )
            vzSig_dic.update( {file : vzSig} )

            vr_dic.update( {file : vr})
            vrSig_dic.update( {file : vRSig})
            spreadB_dic.update( {file : spreadB} )
            rhalf_dic.update( {file : rHalf} )

        plt.title("Forward Velocity vs Flow rate")
        plt.xlabel("Flow [SCCM]")
        plt.ylabel("Forward Velocity [m/s]")
        # plt.errorbar(x=reyn, y=vz, yerr=vzSig, fmt='ro')
        for file in fileList:
            plt.errorbar(x=(fr_dic[file])[0:5], y=(vz_dic[file])[0:5], yerr=(vzSig_dic[file])[0:5], label=legends[file], fmt=formats[file],ls=linestyles[file])
        plt.legend()
        plt.show()
        plt.clf()


        plt.title("Forward Velocity FWHM vs Flow")
        plt.xlabel("Flow [SCCM]")
        plt.ylabel("Velocity FWHM [m/s]")
        # plt.errorbar(x=reyn, y=vzSig, fmt='ro')
        for file in fileList:
            plt.errorbar(x=fr_dic[file], y=2.355*vzSig_dic[file], label=legends[file], fmt=formats[file],ls=linestyles[file])
        plt.legend()
        plt.show()
        plt.clf()

        plt.title("Angular Spread vs Flow")
        plt.xlabel("Flow [SCCM]")
        plt.ylabel("Angular Spread [deg]")
        # plt.errorbar(x=reyn, y=vzSig, fmt='ro')
        for file in fileList:
            plt.errorbar(x=(fr_dic[file])[0:5], y=(spreadB_dic[file])[0:5], label=legends[file], fmt=formats[file],ls=linestyles[file])
        plt.legend()
        plt.show()
        plt.clf()
        
        plt.title("Density FWHM vs Flow\n(Buffer gas 3cm past aperture)")
        plt.xlabel("Flow [SCCM]")
        plt.ylabel("Half Radius [mm]")
        # plt.errorbar(x=reyn, y=vzSig, fmt='ro')
        for file in fileList:
            plt.errorbar(x=(fr_dic[file])[0:7], y=(rhalf_dic[file])[0:7], label=legends[file], fmt=formats[file],ls=linestyles[file])
        plt.legend()
        plt.show()
        plt.clf()

    elif plot==0:

        flowrate = int(which_flow[1:4])
        print(flowrate)

        global fvz, fvr
        vz_field = fvz[which_flow]
        vr_field = fvr[which_flow]

        vzs = np.ones(ARRAY_SIZE)
        vrs = np.ones(ARRAY_SIZE)

        rs = np.sqrt( np.linspace(0, WINDOW_RAD**2, num=ARRAY_SIZE) )
#        angs = np.linspace(0, 2*np.pi, num=ARRAY_SIZE)

        for i in range(ARRAY_SIZE):
            # rs[i] = np.sqrt(np.random.uniform(0, WINDOW_RAD**2))
            # angs[i] = np.random.uniform(0,2*np.pi)

            vzs[i] = vz_field(z, rs[i])[0][0]
            vrs[i] = vr_field(z, rs[i])[0][0]

#        plt.scatter(rs*np.cos(angs), rs*np.sin(angs))
#        plt.axis('equal')
#        plt.show()

        plt.title('Vzs vs Radius')
        plt.scatter(rs, vzs)
        plt.show()

        plt.title('Vrs vs Radius')
        plt.scatter(rs, vrs)
        plt.show()

#Make a polar plot of the velocity field
        
#        RV, THET = np.meshgrid(rs, angs)
#        VZS = np.ones(RV.shape)
#        VRS = np.ones(RV.shape)
#
#        for i in range(ARRAY_SIZE):
#            for j in range(ARRAY_SIZE):
#                r = RV[i,j]
#                VZS[i,j] = vz_field(z, r)[0][0]
#                VRS[i,j] = vr_field(z, r)[0][0]
#
#        #print(dens)
#        if logscale:
#            plt.pcolormesh(RV*np.cos(THET), RV*np.sin(THET), VZS, norm=colors.LogNorm(vmin=VZS.min(),vmax=VZS.max() ) )
#            plt.pcolormesh(RV*np.cos(THET), RV*np.sin(THET), VRS, norm=colors.LogNorm(vmin=VRS.min(),vmax=VRS.max() ) )
#        else:
#            plt.pcolormesh(RV*np.cos(THET), RV*np.sin(THET), VZS)
#            plt.pcolormesh(RV*np.cos(THET), RV*np.sin(THET), VRS)
#
#        plt.axis('equal')
#        plt.show()


        print('Radial velocity: %.1f +- %.1f m/s' %(np.mean(vrs), np.std(vrs)))
        print('Axial velocity: %.1f +- %.1f m/s' %(np.mean(vzs), np.std(vzs)))

        spreadB = 180/np.pi * 2 * np.arctan(np.std(vrs)/np.mean(vzs))
        
        print('Angular spread: %.1f deg'%(spreadB))
        
        rHalf = halfRadius(z=z, which_flow=which_flow)
        
        print('Half radius {}'.format(round(1000*rHalf,3)))
        

        if write == 1:
            with open('/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/BGWindow/{}'.format(file), 'a') as tc:
                tc.write('  '.join(map(str, [z, flowrate, round(np.mean(vzs),3), round(np.std(vzs),3), round(np.mean(vrs),3),\
                         round(np.std(vrs),3), round(spreadB,3), round(1000*rHalf, 3), WINDOW_RAD] ))+'\n')

            tc.close()










# =============================================================================
# Collision number calculations (probably useless)
# =============================================================================

def get_ncoll(z0=0.064, zf=0.120, which_flow='f005'):
    global SIGMA

    z_array = np.linspace(z0, zf, num=SIZE)
    density = np.ones(SIZE)
    for i in range(SIZE):
        density[i] = get_dens(X0, Y0, z_array[i], which_flow)

    prob = SIGMA*density

    ncol = np.trapz(y=prob, x=z_array)
    return ncol

def plot_ncoll(numpoints=100, which_flow='f005'):

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


# =============================================================================
# Outputting final forward velocities for every flow into a DAT file
# =============================================================================

def write_final_vz(write=0):
    folder = '/Users/gabri/Box/HutzlerLab/Data/Woolls_BG_Sims/BGWindow/'
    file = folder+'neckApertureCenterLineVzs.dat'

    if write == 1:
            with open(file, 'a') as tc:
                tc.write('Fr    F     G     H\n')
                for flowrate in ['002', '005', '010', '020', '050', '100', '200']:

                        vzF = get_vz(0, 0, 0.064, 'f'+flowrate)
                        vzG = get_vz(0, 0, 0.064, 'g'+flowrate)
                        vzH = get_vz(0, 0, 0.064, 'h'+flowrate)

                        tc.write('  '.join(map(str, [flowrate, round(vzF,3), round(vzG,3), round(vzH,3)] ))+'\n')

            tc.close()



#if __name__ == '__main__':
#    main()
