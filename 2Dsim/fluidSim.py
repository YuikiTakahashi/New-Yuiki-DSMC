import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

kb = 1.38 * 10**-23
NA = 6.022 * 10**23
T = 4 # Container temperature
m = 0.004 / NA # Ambient gas mass (kg)
M = .190061 / NA # Species mass (kg)
massParam = 2 * m / (m + M)
n = 10**12 # per sq. meter
cross = 4 * (140 * 10**(-12)) # two-helium cross sectional length
cross *= 2 # Rough estimate of He-YbOH cross sectional length

U = kb * T
vMean = 2 * (2 * kb * T / (m * np.pi))**0.5
vMeanM = 2 * (2 * kb * T / (M * np.pi))**0.5

coll_freq = n * cross * vMean
dt = 0.01 / coll_freq # ∆t satisfying E[# collisions in 100∆t] = 1.

# Define a PDF ~ cos(theta) to be used for random determination of impact angle
class angle_pdf(st.rv_continuous):
    def _pdf(self,x):
        return np.cos(x)/2  # Normalized over its range [-pi/2, pi/2]
angle_cv = angle_pdf(a=-np.pi/2, b=np.pi/2, name='angle_pdf') # angle_cv.rvs() for value

# Maxwell-Boltzmann Velocity Distribution
class vel_pdf(st.rv_continuous):
    def _pdf(self,x):
        return (m/(2*np.pi*kb*T))**1.5 * 4*np.pi * x**2 * np.exp(-m*x**2/(2*kb*T))  # x is velocity
vel_cv = vel_pdf(a=0, b=10**4, name='vel_pdf') # vel_cv.rvs() for value

def getData(t1=500, t2=20000, step=500, trials=400, x0=0, y0=0, vx0=0, vy0=0, dvx=0, dvy=0):
    '''
    For a variety of total time steps, determine the expected values of final position,
    square-distance, and path-averaged speed.
    '''
    with open('_'.join(map(str,[t1, t2, step, int(vx0), int(vy0), int(dvx), int(dvy), \
                                int(M/m), int(np.log10(n))]))+'.dat', 'w') as f:
        f.write('   '.join(['time (s)','xAvg (m)','yAvg (m)','SqrAvg (sq. m)',\
                          'SpeedAvg (m/s)','sigX','sigY','sigSqr','sigSpeed'])+'\n')
        for time in range(t1, t2, step):
            print(time)
            xs = []
            ys = []
            squares = []
            speedAvgs = []
            for j in range(trials):
                x, y, vx, vy = x0, y0, vx0, vy0
                speeds = []
                count = 0
                while count < time:
                    if random.random() < 0.01: # 1/100 chance of collision
                        v = vel_cv.rvs() # Maxwell Distribution
                        isoAngle = random.uniform(0, 2*np.pi)
                        impactAngle = angle_cv.rvs() # Random distribution ~ cos(theta) on [-pi/2,pi/2]
                        velParam = ((v * np.cos(isoAngle) + dvx - vx) ** 2 +
                                    (v * np.sin(isoAngle) + dvy - vy) ** 2) ** 0.5
                        isoAngle2 = np.arctan((v * np.sin(isoAngle) + dvy - vy)/(v * np.cos(isoAngle) + dvx - vx))
                        if (v * np.cos(isoAngle) + dvx - vx) < 0:
                            isoAngle2 += np.pi
                        vx += velParam * massParam * np.cos(impactAngle) * np.cos(impactAngle+isoAngle2)
                        vy += velParam * massParam * np.cos(impactAngle) * np.sin(impactAngle+isoAngle2)
                    x += vx * dt
                    y += vy * dt
                    count += 1
                    if count > 0.8 * time:
                        speeds.append((vx**2+vy**2)**0.5)
                speedAvgs.append(np.mean(speeds))
                squares.append(x**2+y**2)
                xs.append(x)
                ys.append(y)
            meanx = str(np.mean(xs))
            meany = str(np.mean(ys))
            meanSq = str(np.mean(squares))
            meanSpeed = str(np.mean(speedAvgs))
            stdx = str(np.std(xs)/(len(xs)**0.5))
            stdy = str(np.std(ys)/(len(ys)**0.5))
            stdSq = str(np.std(squares)/(len(squares)**0.5))
            stdSpeed = str(np.std(speedAvgs)/(len(speedAvgs)**0.5))
            f.write(' '.join([str(time*dt),meanx,meany,meanSq,meanSpeed,stdx,stdy,stdSq,stdSpeed])+'\n')
            print(meanSq+'\n')
    f.close()

def pathTrace(x0=0, y0=0, vx0=0, vy0=0, dvx=0, dvy=0):
    x, y, vx, vy = x0, y0, vx0, vy0
    xs = [0]
    ys = [0]

    # Assume a 10cm x 10cm box
    in_box = True
    while in_box == True:
        # Typically takes few ms to leave box
        if random.random() < 0.01: # 1/100 chance of collision
            v = vel_cv.rvs() # Maxwell Distribution
            #print((2 * U / m)**0.5, (2*kb*T/m)**0.5, v)
            isoAngle = random.uniform(0, 2*np.pi)
            impactAngle = angle_cv.rvs() # Random distribution ~ cos(theta) on [-pi/2,pi/2]
            velParam = ((v * np.cos(isoAngle) + dvx - vx) ** 2 +
                        (v * np.sin(isoAngle) + dvy - vy) ** 2) ** 0.5
            isoAngle2 = np.arctan((v * np.sin(isoAngle) + dvy - vy)/(v * np.cos(isoAngle) + dvx - vx))
            if (v * np.cos(isoAngle) + dvx - vx) < 0:
                isoAngle2 += np.pi
            vx += velParam * massParam * np.cos(impactAngle) * np.cos(impactAngle+isoAngle2)
            vy += velParam * massParam * np.cos(impactAngle) * np.sin(impactAngle+isoAngle2)
        x += vx * dt
        y += vy * dt
        xs.append(x)
        ys.append(y)
        if abs(x) >= 0.05 or abs(y) >= 0.05:
            in_box = False
    print("Iterations to wall: "+str(len(xs)))
    return xs, ys

def getDensityTrend(filename, vel=True):
    global n
    global coll_freq
    global dt
    with open(filename, 'w') as f:
        f.write("n (per sq. m)   mean iterations   mean time to stick   sig iter   sig time\n")
        for n in [10**9, 10**9*5, 10**10, 10**10*5, 10**11, 10**11*5, 10**12]:
            coll_freq = n * cross * vMean
            dt = 0.01 / coll_freq # ∆t satisfying E[# collisions in 100∆t] = 1.
            lens = []
            for j in range(1000):
                if vel == True:
                    xs, ys = pathTrace(vx0=vMeanM)
                else:
                    xs, ys = pathTrace()
                lens.append(len(xs))
            meanLen = str(np.mean(lens))
            meanTime = str(np.mean(lens)*dt)
            stdLen = str(np.std(lens)/(len(lens)**0.5))
            stdTime = str(np.std(lens)*dt/(len(lens)**0.5))
            f.write('%.1E'%n+' '+meanLen+' '+meanTime+' '+stdLen+' '+stdTime+'\n')
    f.close()

def getImage(filename, vel=True):
    for j in range(1, 4):
        if vel == True:
            xs, ys = pathTrace(vx0=vMeanM)
        else:
            xs, ys = pathTrace()
        for i in range(0, len(xs), int(len(xs)/1000)):
            if j == 1 :
                colour = plt.cm.Greens(int(264. * i / len(xs)))
            elif j == 2:
                colour = plt.cm.Blues(int(264. * i / len(xs)))
            else:
                colour = plt.cm.Reds(int(264. * i / len(xs)))
            plt.scatter([xs[i]], [ys[i]], .5, c=colour)

    plt.xlim(-.05, .05)
    plt.ylim(-.05, .05)
    plt.title("Paths of a Light Particle, n = %.0E" %n)
    plt.xlabel("X, meters")
    plt.ylabel("Y, meters")
    plt.savefig(filename)
    plt.show()

getData(vx0 = 0)
getData(vx0 = vMeanM/2)
getData(vx0 = vMeanM)
getData(vx0 = vMeanM*10)

# Relative velocity including VMeanM
# Increased collision probability at high velocity
# Nonconstant flow
# Theory of x(t) distribution by matrix exponentiation?
