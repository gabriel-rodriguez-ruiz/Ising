#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:24:05 2018

@author: Gabriel, Valeria.
"""

# A sweep over temperature

import numpy as np
import matplotlib.pyplot as plt
import ising as ing
import time

#%% User's parameters

Lx = 32 # x size
Ly = 32 # y size
#beta_critico = log(1+sqrt(2))/2 = 0.44069
H = 0
npre = 10000 # amount of steps used to pre-thermalize
nsteps = 1000 # amount of steps used post-thermalization
nplot = 100 # amount of steps between plots

beta_vector = list(np.linspace(0.25, 0.38, 25))
beta_vector = beta_vector + list(np.linspace(0.38, 0.5, 200))
beta_vector = beta_vector + list(np.linspace(0.5, 1, 25))

#%% Initial state

# S is a random matrix whose elements are 1 or -1 (spin projections)
S = ing.initial_condition_2D('hot', (Lx, Ly))

fig = plt.figure()
fig.add_axes()
ax = fig.gca()
ax.pcolormesh(S.T)
plt.title("Estado inicial")
plt.xlabel("X (u.a.)")
plt.ylabel("Y (u.a.)")

#%% Iterate for beta

energy = []
magnetization = []
start = time.time()
for beta in beta_vector:
    
    ##%% Initial state
    S = ing.initial_condition_2D('hot', (Lx, Ly))
    
    ##%% Pre-thermalization
    S = ing.ising_simulation_2D(S, beta, H, printing=False)[0]

    ##%% Post-thermalization
    S, e, m = ing.ising_simulation_2D(S, beta, H, printing=False)
    
    energy.append(list(e))
    magnetization.append(list(m))
stop = time.time()
enlapsed = stop - start
print("Enlapsed time: {:.2f} s".format(enlapsed))

beta_vector = np.array(beta_vector)
energy = np.array(energy)
magnetization = np.array(magnetization)

np.savez("Tcriticas_campo0_8_8.npz", beta_vector=beta_vector, 
         energy_matrix=energy, magnetization_matrix=magnetization, H=H, 
         npre=npre, nsteps=nsteps,
         enlapsed=enlapsed, N = Lx*Ly)
