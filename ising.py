#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:30:44 2018

@author: Gabriel, Valeria
"""

import numpy as np
from math import exp

"""Beware!

This package runs on matrix S[xi, yi]. Meaning...

y

↑
| a b c
| d e f
— — — → x

...should actually be...

|— — — → y
| a d 
| b e
| c f
↓
x

"""

#%%

def immediate_neighbours_2D(shape, indexes):
    """Immediate neighbours of (i,j) on 2D (N, M) matrix with periodic contour.
    
    Parameters
    ----------
    shape : tuple
        Shape of 2D matrix. Each of its elements should be an int: (N, M) where 
        N is the number of total rows and M is the number of total columns.
    indexes : tuple
        Indexes of 2D matrix whose neighbours are required. Each of its 
        elements should be an int: (i, j) where i is the row index and j is the 
        column index.
        
    Returns
    -------
    right : tuple
         Both index of immediate neighbour to the right: (i, j+1)
    up : int, float
         Both index of upper immediate neighbour: (i-1, j)
    left : int, float
         Both index of immediate neighbour to the left: (i, j-1)
    down : int, float
         Both index of lower immediate neighbour: (i+1, j)
    """
    
    N, M = shape
    i, j = indexes
    
    # Find immediate neighbours up and down
    if i != N - 1:
        inext = i + 1
    else: # Periodic contour conditions (N+1 = 1)
        inext = 0
    if i != 0:
        iprevious = i - 1
    else:
        iprevious = N - 1
    up = (iprevious, j)
    down = (inext, j)
    
    
    # Find immediate neighbours up and down
    if j != M - 1:
        jnext = j + 1
    else: # Periodic contour conditions (N+1 = 1)
        jnext = 0
    if j != 0:
        jprevious = j - 1
    else:
        jprevious = M - 1
    right = (i, jnext)
    left = (i, jprevious)
    
    return right, up, left, down

#%%

def all_immediate_neighbours_2D(shape):
    """Returns list of immediate neighbours of a 2D matrix of a certain shape.
    
    Parameters
    ----------
    shape : tuple
        Shape of 2D matrix. Each of its elements should be an int: (N, M) where 
        N is the number of total rows and M is the number of total columns.
    
    Returns
    ------
    neighbours : list
        List of all immediate neighbours. It's a N*M-length list. Each of 
        its elements is a tuple containing the two index of 4 immediate 
        neighbours: right, up, left, down.
    """
    
    N, M = shape
    neighbours = []
    
    for i in range(N):
        for j in range(M):
            neighbours.append(immediate_neighbours_2D(shape, (i, j)))
        
    return neighbours

#%%

def energy_2D(S):
    """Returns energy from a spin matrix S.
    
    Parameters
    ----------
    S : np.array
        A 2D matrix with N rows (up-down i index) and M columns 
        (right-left j index).
    
    Returns
    -------
    En : float
        The energy of that specific spin configuration.
    """
        
    En = 0
    
    neighbours = all_immediate_neighbours_2D(S.shape)
    
    for Sij, nij in zip(np.reshape(S, S.size), neighbours):
        En = En - Sij * sum([S[index] for index in nij])

    # Since each pair of spins was counted twice...
    En = En/2

    return En

#%%

def ising_step_2D(S, beta, neighbours, p):
    """Executes one step in the Markov chain using Metropolis algorithm.
    
    Parameters
    ----------
    S : np.array
        A 2D matrix with N rows (up-down xi index) and M columns 
        (right-left yi index).
    beta : float
        The multiplicative inverse of the Temperature of the system.
    neighbours : list
        The list of immediate neighbours on each place of a flatten matrix 
        of same shape as S. Each of its elements should be an iterative wich 
        holds the index of the 4 immediate neighbours.
        
    Returns
    -------
    new_S : np.array
        A 2D matrix with N rows (up-down xi index) and M columns 
        (right-left yi index).
    dE : np.array
        Accumulated energy change.
    dM : np.array
        Accumulated magnetization change.
    """
    
    dE = 0  
    dM = 0
#    new_S = []
    new_S = np.reshape(S, S.size)
    k = 0
    # For each place in the matrix S, a random spin flip will be proposed.
    for Sij, nij, pk in zip(np.reshape(S, S.size), neighbours, p):
        
        # Partial energy difference
        dE_partial = 2 * Sij * sum([S[index] for index in nij])
        # -S is the proposed new spin
        
        if dE_partial<0:
            # If energy decreases, spin flip will be accepted.
#            new_S.append(-Sij)
            new_S[k] = -Sij
            dE = dE + dE_partial
            dM = dM - 2*Sij # new_spin - old_spin
            
        else:
            # If energy increases, the change will be considered...
#            p = np.random.rand()
            expbetaE = exp(-beta * dE_partial)
            # It will only be accepted with probability exp(-beta*dE)
            if pk < expbetaE:
#                new_S.append(-Sij)
                new_S[k] = -Sij
                dE = dE + dE_partial
                dM = dM - 2*Sij
#            else:
#                new_S.append(Sij)
        
        k += 1
    
#    new_S = np.array(new_S)
    new_S = np.reshape(new_S, S.shape)
                    
    return new_S, dE, dM

"""
¿Viste que spin por spin probás ver si hay que invertirlo? Ahí gran parte de la 
decisión recae sobre los vecinos de ese spin. 

Ponele que tenés una matriz S. Recorrés dos spines y el segundo lo das vuelta, 
por lo cual la matriz S' ahora tiene un spin invertido respecto a S. 

Ahora vas al tercero. ¿Ahora mirás los vecinos de ese spin en S? ¿O en S'?

--> El código con array se fija en los vecinos de S'.
--> El código con listas se fija en los vecinos de S.

"""

#%%
    
def ising_step_2D_list(S, beta, neighbours, p):
    """Executes one step in the Markov chain using Metropolis algorithm.
    
    Parameters
    ----------
    S : np.array
        A 2D matrix with N rows (up-down xi index) and M columns 
        (right-left yi index).
    beta : float
        The multiplicative inverse of the Temperature of the system.
    neighbours : list
        The list of immediate neighbours on each place of a flatten matrix 
        of same shape as S. Each of its elements should be an iterative wich 
        holds the index of the 4 immediate neighbours.
        
    Returns
    -------
    new_S : np.array
        A 2D matrix with N rows (up-down xi index) and M columns 
        (right-left yi index).
    dE : np.array
        Accumulated energy change.
    dM : np.array
        Accumulated magnetization change.
    """
    
    dE = 0  
    dM = 0
    new_S = []
#    new_S = np.reshape(S, S.size)
#    k = 0
    # For each place in the matrix S, a random spin flip will be proposed.
    for Sij, nij, pk in zip(np.reshape(S, S.size), neighbours, p):
        
        # Partial energy difference
        dE_partial = 2 * Sij * sum([S[index] for index in nij])
        # -S is the proposed new spin
        
        if dE_partial<0:
            # If energy decreases, spin flip will be accepted.
            new_S.append(-Sij)
#            new_S[k] = -Sij
            dE = dE + dE_partial
            dM = dM - 2*Sij # new_spin - old_spin
            
        else:
            # If energy increases, the change will be considered...
#            p = np.random.rand()
            expbetaE = exp(-beta * dE_partial)
            # It will only be accepted with probability exp(-beta*dE)
            if pk < expbetaE:
                new_S.append(-Sij)
#                new_S[k] = -Sij
                dE = dE + dE_partial
                dM = dM - 2*Sij
            else:
                new_S.append(Sij)
        
#        k += 1
    
#    new_S = np.array(new_S)
    new_S = np.reshape(new_S, S.shape)
                    
    return new_S, dE, dM

#%%

def ising_simulation_2D(S, beta, nsteps=1000):
    """Executes several steps in a Markov chain using Metropolis algorithm.
    
    Parameters
    ----------
    S : np.array
        2D matrix with N rows (up-down xi index) and M columns (right-left yi 
        index).
    beta : float
        Multiplicative inverse of the temperature of the system.
    nsteps=1000 : int, optional
        Desired amount of steps.
        
    Returns
    -------
    new_S : np.array
        Final 2D matrix. Same as S, it has N rows (up-down xi index) and M 
        columns (right-left yi index).
    E : np.array
        Accumulated energy change array. Holds one value per step.
    M : np.array
        Accumulated magnetization change array. Holds one value per step.
    """
    
    neighbours = all_immediate_neighbours_2D(S.shape)

    energy = []
    magnetization = []
    
    energy.append(energy_2D(S))
    magnetization.append(np.sum(S))
    
    print("Running...")
    for n in range(nsteps):
        S, dE, dM = ising_step_2D(S, beta, neighbours)
        energy.append(energy[-1] + dE)
        magnetization.append(magnetization[-1] + dM)
    print("Done running :)")
    
    energy = np.array(energy)
    magnetization = np.array(magnetization)
                    
    return S, energy, magnetization

#%%

def initial_condition_2D(condition, shape):
    """Returns the initial spin matrix S depending on condition.
    
    Parameters
    ----------
    condition : string
        Initial conditions' descriptor. Must be either 'hot' (random 
        initial condition) or 'cold' (uniform initial condition full of 
        parallel spins)
    shape : tuple of ints
        Spin matrix dimensions. Each of its elements should be an int: 
        (N, M) where N is the amount of rows (x-axis) and M is the 
        amount of columns (y-axis).
    """
    
    if condition == "hot":
        S = np.where(np.random.rand(*shape) > 0.5, 1, -1)        
    elif condition == "cold":
        S = np.ones(shape)
    else:
        raise TypeError("Wrong condition. Must be 'hot' or 'cold'")
    
    S = np.array(S, dtype='int8')
    
    return S