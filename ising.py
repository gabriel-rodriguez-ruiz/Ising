#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:30:44 2018

@author: Gabriel, Valeria
"""

import numpy as np

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

def immediate_neighbours_2D(S, i, j):
    """Returns 4 immediate neighbours of S[i,j] with periodic contour.
    
    Parameters
    ----------
    S : np.array
        A 2D matrix with N rows (up-down xi index) and M columns 
        (right-left yi index).
    
    Returns
    -------
    right : int, float
         Value of immediate neighbour to the right: S[i, j+1]
    up : int, float
         Value of upper immediate neighbour: S[i-1, j]
    left : int, float
         Value of immediate neighbour to the left: S[i, j-1]
    down : int, float
         Value of lower immediate neighbour: S[i+1, j]
        
    Example
    -------
    >> import numpy as np
    >> S = np.reshape(range(1,10), (3,3))
    >> S
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    >> S[1, 1]
    5
    >> immediate_neighbours(S, 1, 1)
    (6, 2, 4, 8)
    >> S[1, 0]
    4
    >> immediate_neighbours(S, 1, 0)
    (5, 1, 6, 7)
    """
    
    N, M = np.shape(S)
    
    # Find immediate neighbours up and down
    if i != N - 1:
        inext = i + 1
    else: # Periodic contour conditions (N+1 = 1)
        inext = 0
    if i != 0:
        iprevious = i - 1
    else:
        iprevious = N - 1
    up = S[iprevious, j]
    down = S[inext, j]
    
    
    # Find immediate neighbours up and down
    if j != M - 1:
        jnext = j + 1
    else: # Periodic contour conditions (N+1 = 1)
        jnext = 0
    if j != 0:
        jprevious = j - 1
    else:
        jprevious = M - 1
    right = S[i, jnext]
    left = S[i, jprevious]
    
    return right, up, left, down

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
    
    N, M = np.shape(S)
    
    En = 0
    for i in range(N):
        for j in range(M):
            
            # Find immediate neighbours' coordinates
            neighbours = immediate_neighbours_2D(S, i, j)
            
            # Now make an addition to energy
            En = En - S[i,j] * sum(neighbours)

    # Since each pair of spins was counted twice...
    En = En/2

    return En

#%%

def ising_step_2D(S, beta, H=0):
    """Executes one step in the Markov chain using Metropolis algorithm.
    
    Parameters
    ----------
    S : np.array
        A 2D matrix with N rows (up-down xi index) and M columns 
        (right-left yi index).
    beta : float
        The multiplicative inverse of the Temperature of the system.
    
    H : float between -1 and 1 (optional)
        External magnetic field.
            
    Returns
    -------
    S : np.array
        A 2D matrix with N rows (up-down xi index) and M columns 
        (right-left yi index).
    dE : np.array
        Accumulated energy change.
    dM : np.array
        Accumulated magnetization change.
    """
    
    N, M = np.shape(S)
    dE = 0  
    dM = 0 
    
    #For each place in the matrix S, a random spin flip will be proposed. 
    for i in range(N):
        for j in range(M):
            
            spin_old = S[i, j]
            spin_new = -S[i, j] # spin flip
            
            spin_neighbours = immediate_neighbours_2D(S, i, j)
            
            #Partial energy difference, meaning the difference between the energy after spin flip and before.
            dE_partial = 2 * spin_old * sum(spin_neighbours) + 2 * spin_old *  H 
            
            if dE_partial<0:
                #If energy decreases, spin flip will be accepted.
                S[i, j] = spin_new
                dE = dE + dE_partial
                dM = dM + (spin_new - spin_old)
                
            else:
                #If energy increases, the change will be considered...
                p = np.random.rand()
                expbetaE = np.exp(-beta * dE_partial)
                
                # Only if a random number is below exp(-beta*dE_partial)...
                if p < expbetaE:
                    # ...change will be accepted.
                    # Meaning probability will be exp(-beta*dE_partial)
                    S[i, j] = spin_new
                    dE = dE + dE_partial
                    dM = dM + (spin_new - spin_old)
                    
    return S, dE, dM

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
    
    return S