#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:24:05 2018

@author: Gabriel, Valeria.
"""

# General script for a run

import numpy as np
#import matplotlib.axes
import ising as ing
import time

##%% User's parameters

start = time.time()

    

Lx = 32 # x size
Ly = 32 # y size
#beta_critico = log(1+sqrt(2))/2 = 0.44069
#beta = 1 # 1/kT
H = 0
npre = 10000 # amount of steps used to pre-thermalize
nsteps = 1000 # amount of steps used post-thermalization
nplot = 100 # amount of steps between plots

beta_points = list(np.linspace(0.25, 0.38, 25)) + list(np.linspace(0.38, 0.5, 200)) + list(np.linspace(0.5, 1, 25))
#beta_points = list(np.linspace(0.42, 0.46, 100))
beta_vector = np.array(beta_points)
energy_matrix = np.zeros((len(beta_points), nsteps + 1))
magnetization_matrix = np.zeros((len(beta_points), nsteps + 1))

for i in range(len(beta_points)):
   
    beta = beta_points[i]
    ##%% Initial state
    
    # S is a random matrix whose elements are 1 or -1 (spin projections)
    S = ing.initial_condition_2D('hot', (Lx, Ly))
    energy = np.zeros(nsteps + 1)
    magnetization = np.zeros(nsteps + 1)
    """
    fig = plt.figure()
    fig.add_axes()
    ax = fig.gca()
    ax.pcolormesh(S.T)
    plt.title("Estado inicial")
    plt.xlabel("X (u.a.)")
    plt.ylabel("Y (u.a.)")
    """
    ##%% Pre-thermalization
    
    #start = time.time()

    for n in range(npre):
        S = ing.ising_step_2D(S, beta, H)[0]
        #    print("Step: {:.0f}/{:.0f}".format(n+1, npre))
        
        energy[0] = ing.energy_2D(S)
        magnetization[0] = np.sum(S)
        
        """
        stop = time.time()
        enlapsed = stop - start
        print("Enlapsed time: {:.2f} s".format(enlapsed))
        print("Current energy: {:.2f}".format(energy[0]))
        print("Current magnetization: {:.2f}".format(magnetization[0]))
        """
        """
        fig = plt.figure()
        fig.add_axes()
        ax = fig.gca()
        ax.pcolormesh(S.T)
        plt.title("Estado pretermalizado")
        plt.xlabel("X (u.a.)")
        plt.ylabel("Y (u.a.)")
        """
    ##%% Post-thermalization
    
    #start = time.time()
    
    for n in range(nsteps):
        S, dE, dM = ing.ising_step_2D(S, beta, H)
        energy[n + 1] = energy[n] + dE
        magnetization[n + 1] = magnetization[n] + dM
        #print("Step: {:.0f}/{:.0f}".format(n+1, nsteps))
        
    energy_matrix[i,:] = energy
    magnetization_matrix[i, :] = magnetization
        
stop = time.time()
enlapsed = stop - start
print("Enlapsed time: {:.2f} s".format(enlapsed))
    
"""
print("Current energy: {:.2f}".format(energy[-1]))
print("Current magnetization: {:.2f}".format(magnetization[-1]))
"""
"""
fig = plt.figure()
fig.add_axes()
ax = fig.gca()
ax.pcolormesh(S.T)
plt.title("Estado final")
plt.xlabel("X (u.a.)")
plt.ylabel("Y (u.a.)")
"""
##%% Some general plots

"""    
if n%nshow==0:
ax.pcolormesh(S)
plt.title("T = {}".format(1/beta)) #llos graficos no funcionan


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

"""

np.savez("Tcriticas_campo0_8_8.npz", beta_vector=beta_vector, energy_matrix=energy_matrix, magnetization_matrix=magnetization_matrix, H=H, npre=npre, nsteps=nsteps,
       enlapsed=enlapsed, N = Lx*Ly)
