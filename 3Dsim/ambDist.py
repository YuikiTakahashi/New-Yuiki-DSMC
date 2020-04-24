#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:18:26 2020

@author: dave
"""


import numpy as np
import emcee
import matplotlib.pyplot as plt
from constants import *  

def log_prob(u, relFlow, T):
    # relFlow = v_YbOH - Flow is a 3-vector.
    r = u[0]
    theta = u[1]
    phi = u[2]
    if (r < 0 or theta < 0 or theta > np.pi or phi < 0 or phi > 2 * np.pi):
        return np.log(0)
    uz = r * np.cos(theta)
    ux = r * np.sin(theta) * np.cos(phi)
    uy = r * np.sin(theta) * np.sin(phi)
    MB = (M_HE/(2*np.pi*KB*T))**1.5 * r**2 * 4*np.pi * np.exp(-M_HE*r**2/(2*KB*T))
    polar = np.sin(theta)
    relx = ux - relFlow[0]
    rely = uy - relFlow[1]
    relz = uz - relFlow[2]
    rel = np.sqrt(relx**2 + rely**2 + relz**2)
    return np.log(MB * polar * rel)

def log_prob_approx(u, relFlow, T):
    '''
    THIS IS ONLY HERE FOR TESTING / COMPARISON - DO NOT USE OTHERWISE.
    '''
    r = u[0]
    theta = u[1]
    phi = u[2]
    if (r < 0 or theta < 0 or theta > np.pi or phi < 0 or phi > 2 * np.pi):
        return np.log(0)
    MB = (M_HE/(2*np.pi*KB*T))**1.5 * r**2 * 4*np.pi * np.exp(-M_HE*r**2/(2*KB*T))
    polar = np.sin(theta)
    return np.log(MB * polar * r)

def get_samples(f, rel, T):
    ndim = 3
    nwalkers = 32
    p0u = np.random.rand(nwalkers) * 200
    p0t = np.random.rand(nwalkers) * np.pi
    p0p = np.random.rand(nwalkers) * 2 * np.pi
    p0 = np.array([p0u.tolist(), p0t.tolist(), p0p.tolist()]).transpose()
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, f, args=[rel, T])
    
    state = sampler.run_mcmc(p0, 1000)
    sampler.reset()
    
    sampler.run_mcmc(state, 10000);
    return sampler.get_chain(flat=True)

def get_sample(f, rel, T, nwalkers=8, burn=400, n=1):
    '''
    Uses emcee to spit out a (u, theta, phi) ambient velocity vector.
    Density plot tests showed that 400 burns is plenty. Can reduce to 100 if needed for speed.
    Keep nwalkers > 2 * dim = 6.
    '''
    ndim = 3
    p0u = np.random.rand(nwalkers) * 200
    p0t = np.random.rand(nwalkers) * np.pi
    p0p = np.random.rand(nwalkers) * 2 * np.pi
    p0 = np.array([p0u.tolist(), p0t.tolist(), p0p.tolist()]).transpose()
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, f, args=[rel, T])
    
    state = sampler.run_mcmc(p0, burn)
    sampler.reset()
    
    sampler.run_mcmc(state, n);
    return sampler.get_chain(flat=True)[-1]

def compare_probs(rel, T):
    '''
    Compares probability distributions arising from exact vs. approximate methods.
    '''
    samples = get_samples(log_prob, rel, T)
    samples_approx = get_samples(log_prob_approx, rel, T)
    labels = ["Speed (m/s)", "Theta", "Phi"]
    for i in range(3):
        plt.figure()
        plt.hist(samples[:, i], 200, color="r", histtype="step", cumulative=True, label="Exact")
        plt.hist(samples_approx[:, i], 200, color="b", histtype="step", cumulative=True, label="Approx")
        if i == 0:
            plt.xlim(0, 300)
        plt.xlabel(labels[i])
        plt.gca().set_yticks([]);
        plt.legend(loc=2)
        plt.title("(%d, %d, %d) m/s YbOH Velocity Relative to Flow"%(rel[0], rel[1], rel[2]))
        