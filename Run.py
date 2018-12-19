#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:24:05 2018

@author: Gabriel, Valeria.
"""

# General script for a run

import numpy as np
import matplotlib.pyplot as plt
import ising as ing

#%% User's parameters

Lx = 15 # x size
Ly = 15 # y size
beta = 0.9 # 1/kT
nsteps = 2000 # amount of steps
nplots = 4
animation = True
full = False
save = True

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

#%% One Run

if not animation:
    Sf, energy, magnetization = ing.ising_simulation_2D(S, beta, nsteps=nsteps)
else:
    results = ing.ising_animation_2D(S, beta, nsteps=nsteps, 
                                     nplots=nplots, full=full, save=save)

#%% Final State
    
"""
Watch out! Animation only fills 'results' with data once the simulation is over
"""
    
if animation:
    Sf = results['Sf']
    energy = results['energy']
    magnetization = results['magnetization']

fig = plt.figure()
fig.add_axes()
ax = fig.gca()
ax.pcolormesh(Sf.T)
plt.title("Estado final")
plt.xlabel("X (u.a.)")
plt.ylabel("Y (u.a.)")

fig = plt.figure()
fig.add_axes()
ax = fig.gca()
ax.pcolormesh(abs(Sf.T-S.T))
plt.title("Diferencia")
plt.xlabel("X (u.a.)")
plt.ylabel("Y (u.a.)")

#%% Some general plots

"""    
    if n%nshow==0:
        ax.pcolormesh(S)
        plt.title("T = {}".format(1/beta)) #llos graficos no funcionan
"""

plt.figure()
plt.subplot(2,1,1)
plt.plot(energy, "o-")
plt.ylabel("Energía")

plt.subplot(2,1,2)
plt.plot(magnetization, "o-")
plt.ylabel("Magnetización")
plt.xlabel("Paso")

n = np.arange(1, len(energy)+1)
mean_energy_accumulated = np.cumsum(energy)/n
mean_magnetization_accumulated = np.cumsum(magnetization)/n

plt.figure()
plt.subplot(2,1,1)
plt.plot(mean_energy_accumulated, "o-")
plt.ylabel("Energía media")

plt.subplot(2,1,2)
plt.plot(mean_magnetization_accumulated, "o-")
plt.ylabel("Magnetización media")
plt.xlabel("Paso")
