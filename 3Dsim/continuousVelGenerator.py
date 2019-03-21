# Use marginal distributions of coll_vel_pdf(x, y, z) to randomly determine x, y, z.
class vel_x_pdf(st.rv_continuous):
    def _pdf(self,x):
        global vel_x_norm
        if vel_x_norm == 0:
            vel_x_norm = integrate.tplquad(lambda z, y, x: coll_vel_pdf(x, y, z), -vMean*4, vMean*4, \
                    lambda y: -vMean*4, lambda y: vMean*4, lambda x,y:-vMean*4, lambda x,y:vMean*4)[0]
        return integrate.dblquad(lambda z, y: coll_vel_pdf(x, y, z), \
                                 -vMean*4, vMean*4, lambda y: -vMean*4, \
                                 lambda y: vMean*4)[0]/vel_x_norm
vel_x_cv = vel_x_pdf(a=-vMean*4, b=vMean*4, name='vel_x_pdf') # vel_x_cv.rvs() for value

class vel_y_pdf(st.rv_continuous):
    def _pdf(self,y):
        global vel_y_norm
        if vel_y_norm == 0:
            vel_y_norm = integrate.dblquad(lambda z, y: coll_vel_pdf(vel_x, y, z), -vMean*4, vMean*4, \
                                 lambda y: -vMean*4, lambda y: vMean*4)[0]
        return integrate.quad(lambda z: coll_vel_pdf(vel_x, y, z), -vMean*4, vMean*4)[0] / vel_y_norm
vel_y_cv = vel_y_pdf(a=-vMean*4, b=vMean*4, name='vel_y_pdf') # vel_y_cv.rvs() for value

class vel_z_pdf(st.rv_continuous):
    def _pdf(self,z):
        global vel_z_norm
        if vel_z_norm == 0:
            vel_z_norm = integrate.quad(lambda z: coll_vel_pdf(vel_x, vel_y, z), -vMean*4, vMean*4)[0]
        return coll_vel_pdf(vel_x, vel_y, z) / vel_z_norm
vel_z_cv = vel_z_pdf(a=-vMean*4, b=vMean*4, name='vel_z_pdf') # vel_z_cv.rvs() for value
def getAmbientVelocity():
    global vel_x, vel_y, vel_z # Allows vel_y_cv, vel_z_cv to know vel_x, vel_y
    global vel_x_norm, vel_y_norm, vel_z_norm
    vel_x_norm, vel_y_norm, vel_z_norm = 0, 0, 0
    # Get thermal velocity components
    vel_x = vel_x_cv.rvs()
    vel_y = vel_y_cv.rvs()
    vel_z = vel_z_cv.rvs()
    return vel_x + xFlow - vx, vel_y + yFlow - vy, vel_z + zFlow - vz