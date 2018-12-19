#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:17:36 2018

@author: Gabriel, Valeria
"""

import ising as ing
import numpy as np
import matplotlib.pyplot as plt

#%%

# PARAMETERS

Lx = 8 # x size
Ly = 8 # y size
#beta_critico = log(1+sqrt(2))/2 = 0.44069
beta = 0.441

npasos = 10000   # total number of steps
# 100000 steps take 12 min for a 15x15 grid
# 2000 steps take 14.5 seg for the same grid

# ACTIVE CODE

# Initial random condition
S = ing.initial_condition_2D('hot', (Lx, Ly))

# Ising via Metropolis execution
Sf, energy, magnetization = ing.ising_simulation_2D(S, beta, nsteps=npasos-1)

# Mean parameters
n = np.arange(1, npasos + 1, 1)
mean_energy = np.cumsum(energy)/n
mean_magnetization = np.cumsum(magnetization)/n

# Energy plot
plt.figure("Energía promedio")
plt.plot(mean_energy,"o")
plt.ylabel("Energía promedio")
plt.xlabel("Paso")

# Magnetization plot
plt.figure("Magnetización promedio")
plt.plot(mean_magnetization,"o")
plt.ylabel("Magnetización promedio")
plt.xlabel("Paso")

plt.figure("Magnetización", clear=True)
plt.plot(magnetization/(Lx*Ly),"o", markersize=2)
plt.ylabel("Magnetización (u. a.)")
plt.xlabel("Paso")

"""Conclutions

#Grilla de 15x15
# High temperatures ==> npre=5000
# Temperatures close to Tc ==> npre=50000
# Low temperatures ==> npre=10000
"""