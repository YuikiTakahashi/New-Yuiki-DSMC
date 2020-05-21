#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:13:37 2020

@author: dave
"""

# =============================================================================
# Update these three lines for the collision frequency correction
# =============================================================================

# Replace
dt, no_collide = self.get_derived_quants(temp=bgTemp, dens=bgDens, prev_dt=dt
# With
dt, no_collide = self.get_derived_quants(temp=bgTemp, dens=bgDens, prev_dt=dt, \
                                                         v_rel=np.linalg.norm((bgFlow-np.array([vx, vy, vz]))))
    
# Replace
def get_derived_quants(self, temp, dens, prev_dt):
# With
def get_derived_quants(self, temp, dens, prev_dt, v_rel):
    
# Replace
coll_freq = n * cross * v_mean
# With   
coll_freq = n * cross * (v_mean + 4/(3*np.pi) * v_rel**2/v_mean)



# =============================================================================
# Insert this code with the other correc options before the "else" statement.
# =============================================================================

        elif correc == 3:
            '''
            An approximate implementation of Bayesian-biased velocities,
            adjusting not only the speed probability distribution but also
            the angle of incidence in accordance with the relative flow of
            helium relative to the molecule.
            '''
            # First, we calculate the result in a coordinate system whose
            # z-axis is aligned with v_rel = v_flow - v_mol.
            relx = xFlow - vx
            rely = yFlow - vy
            relz = zFlow - vz
            v_rel = np.sqrt(relx**2 + rely**2 + relz**2)
            u = vel3_cv.rvs(T=temp, v=v_rel)
            theta = theta3_cv.rvs(T=temp, v=v_rel, u=u)
            
            # Now, we generate a random phi in that coordinate system
            # and convert the result into our original coordinates.
            ux, uy, uz = convert_coords(u, theta, relx, rely, relz)
            
            # Finally, we use this thermal result to return the net helium
            # velocity in the molecule's rest frame.
            return ux + xFlow - vx, uy + yFlow - vy, uz + zFlow - vz


# =============================================================================
# Add these definitions with all the other PDF's at the bottom of the code.
# =============================================================================

class vel3_pdf(st.rv_continuous):
    def _pdf(self, u, T, v):
        maxwell = u**2 * np.exp(-M_HE*u**2/(2*KB*T))
        bayes = u + v**2 / (3*u)
        norm = KB*T * (6*KB*T + M_HE * v**2)/(3 * M_HE**2)
        return maxwell * bayes / norm
vel3_cv = vel3_pdf(a=0, b=np.inf, name='vel3_pdf') # vel3_cv.rvs(T=T, v=v) for value

class theta3_pdf(st.rv_continuous):
    def _pdf(self, x, T, v, u):
        maxwell = u**2 * np.exp(-M_HE*u**2/(2*KB*T))
        bayes = u * np.sin(x) * (1 + np.cos(x) * v/u + (np.sin(x)*v/u)**2 / 2)
        norm = 2/3 * np.exp(-M_HE * u**2 / (2*KB*T)) * u * (3*u**2 + v**2)
        return maxwell * bayes / norm  # Normalized over its range [0, pi]
theta3_cv = theta3_pdf(a=0, b=np.pi, name='theta3_pdf') # theta3_cv.rvs(T=T, v=v, u=u) for value

def convert_coords(vel3, theta3, x, y, z):
    # (x, y, z) = vflow - vmol
    r = np.sqrt(x**2 + y**2 + z**2)
    phi3 = np.random.random() * 2 * np.pi
    vx3 = vel3 * np.sin(theta3) * np.cos(phi3)
    vy3 = vel3 * np.sin(theta3) * np.sin(phi3)
    vz3 = vel3 * np.cos(theta3)
    
    vx = (y**2 + x**2*z/r)/(x**2 + y**2) * vx3 + \
        (z/r - 1)*x*y/(x**2 + y**2) * vy3 + \
        x/r * vz3
    
    vy = (z/r - 1)*x*y/(x**2 + y**2) * vx3 + \
        (x**2 + y**2*z/r)/(x**2 + y**2) * vy3 + \
        y/r * vz3
        
    vz = -x/r * vx3 - y/r * vy3 + z/r * vz3
    
    return (vx, vy, vz)