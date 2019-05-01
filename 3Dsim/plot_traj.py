#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

X_SIZE = 256
Y_SIZE = 128

x_data = []
y_data = []

fileobj = open('NewDataH/try_1_succ.dat')
next(fileobj)

for line in fileobj:
    row = line.split()
    x_data.append(float(row[0]))
    y_data.append(float(row[1]))

x = np.array(x_data)
y = np.array(y_data)


#u = np.array(vx_data)
#v = np.array(vy_data)
#vorticity = np.array(vorticity_data)

#u = u.reshape((X_SIZE,Y_SIZE))
#vorticity = vorticity.reshape((X_SIZE,Y_SIZE))


#x = np.arange(-10,10,1)
#y = np.arange(-10,10,1)
#x = [0.5, 0, -0.5, 0, 0.5, 1, 1.5]
#y = [0.25, -0.25, 0.25, 0, 0.5, 0.25, -0.25]
#u, v = np.meshgrid(x,y)

#fig, ax = plt.subplots()

l = plt.plot(x, y, 'bo')
plt.setp(l, markersize = 2)
plt.setp(l, markerfacecolor = 'None')

plt.show()
#imgshow = plt.imshow(u)
#imgshow = plt.imshow(vorticity)

#plt.colorbar()
#plt.show()


'''
ax.imshow(u)
#plt.colorbar()
plt.show()

q = ax.quiver(x,y,u,v)
ax.quiverkey(q, X = 0.3, Y = 1.1, U = 0.01,label = 'Quiver key, length 10', labelpos = 'E')
plt.show()
'''
