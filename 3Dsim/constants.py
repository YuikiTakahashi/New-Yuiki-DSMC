#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:21:39 2020

@author: dave
"""
import numpy as np
KB = 1.38 * 10**-23 #Boltzmann
NA = 6.022 * 10**23 #Avogadro
M_HE = 0.004 / NA # Helium gas particle mass (kg)
M_S = .190061 / NA # Species mass (kg)
M_RED = (M_HE * M_S) / (M_HE + M_S)
MASS_PARAM = 2 * M_HE / (M_HE + M_S)
CROSS_HE = 4 * np.pi * (140 * 10**(-12))**2 # helium-helium cross section
#CROSS_YBF = 20 * 4 * np.pi * (140 * 10**(-12))**2 # helium-helium cross section
CROSS_YBF = 5*100*10**(-20) #YbF-He cross section at 20 K (m^2) 
# YbOH mass in amu
mass = 173 + 16 + 1  
# buffer gass mass (He) in amu
bgmass = 4