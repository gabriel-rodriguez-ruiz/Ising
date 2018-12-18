#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:30:44 2018

@author: Gabriel, Valeria
"""

import threading
import queue
import numpy as np
from math import exp
import matplotlib.pyplot as plt

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

def flat_matrix_index_conversion_2D(shape):
    """Returns both 2D index converters flat-to-matrix and matrix-to-flat.
    
    Parameters
    ----------
    shape : tuple
        Matrix dimensions. Must hold two int values: (N, M) where N is the 
        number of rows and M is the number of columns.
    
    Returns
    -------
    flat_to_matrix : function
        Flat-to-matrix index converter. Takes in an int and returns a tuple;
        i.e.: flat_to_matrix(0) = (0,0)
    flat_to_matrix : function
        Matrix-to-flat index converter. Takes in a tuple and returns an int; 
        i.e.: matrix_to_flat((0,0)) = 0
    """
    
    matrix_index = []
    flat_index = {}
    for i in range(shape[0]):
        for j in range(shape[1]):
            flat_index.update({(i,j) : len(matrix_index)})
            matrix_index.append((i,j))
    
    flat_to_matrix = lambda i : matrix_index[i]
    matrix_to_flat = lambda tup : flat_index[tup]
    
    return flat_to_matrix, matrix_to_flat
    
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

def ising_step_2D(S, beta, H, neighbours, p):
    """Executes one matrix step in the Markov chain using Metropolis algorithm.
    
    Parameters
    ----------
    S : np.array
        A flatten 2D spin matrix. Each of its elements should be +1 or -1.
    beta : float
        The multiplicative inverse of the temperature of the system.
    H : float
        External magnetic field. Must be between 1 and -1.
    neighbours : list
        The list of immediate neighbours on each place of the flatten matrix. 
        Each of its elements should be an iterative wich holds the index of the 
        4 immediate neighbours.
        
    Returns
    -------
    S : np.array
        New flatten 2D spin matrix. Each of its elements is +1 or -1.
    dE : np.array
        Accumulated energy change.
    dM : np.array
        Accumulated magnetization change.
    """
    
    dE = 0  
    dM = 0
    k = 0
    # For each place in the matrix S, a random spin flip will be proposed.
    for Sij, nij, pk in zip(S, neighbours, p):
        
        # Partial energy difference
        dE_partial = 2 * Sij * ( sum([S[index] for index in nij]) +  H )
        # -S is the proposed new spin
        
        if dE_partial<0:
            # If energy decreases, spin flip will be accepted.
            S[k] = -Sij
            dE = dE + dE_partial
            dM = dM - 2*Sij # new_spin - old_spin
            
        else:
            # If energy increases, the change will be considered...
#            p = np.random.rand()
            expbetaE = exp(-beta * dE_partial)
            # It will only be accepted with probability exp(-beta*dE_partial)
            if pk < expbetaE:
                S[k] = -Sij
                dE = dE + dE_partial
                dM = dM - 2*Sij
        
        k += 1
                    
    return S, dE, dM

#%%

def ising_simulation_2D(S, beta, H=0, nsteps=1000, 
                        animation=False, nplot=200):
    """Executes several steps in a Markov chain using Metropolis algorithm.
    
    Parameters
    ----------
    S : np.array
        2D spin matrix. Each of its elements should be +1 or -1.
    beta : float
        Multiplicative inverse of the temperature of the system.
    H : float
        External magnetic field. Must be between 1 and -1.
    nsteps=1000 : int, optional
        Desired amount of steps.
        
    Returns
    -------
    Sf : np.array
        Final 2D spin matrix. Each of its elements are +1 or -1.
    E : np.array
        Accumulated energy change array. Holds one value per step.
    M : np.array
        Accumulated magnetization change array. Holds one value per step.
    """
    
    shape = S.shape
    neighbours = all_immediate_neighbours_2D(shape)
    matrix_to_flat = flat_matrix_index_conversion_2D(shape)[1]
    flat_neighbours = []
    for nij in neighbours:
        one_element = [matrix_to_flat(index) for index in nij]
        flat_neighbours.append(tuple(one_element))
    neighbours = flat_neighbours

    energy = []
    magnetization = []
    
#    if animation:
#        generator = ising_animation_2D(
#                S, 
#                beta, 
#                H, 
#                textlabel=lambda i : 'Paso {:.0f}'.format(i*nplot))
#        q = queue.Queue()
#        def worker():
#            while True:
#                S = q.get() # Waits for data to be available
#                generator.send(S)
#                next(generator)
#                plt.show()
#        t = threading.Thread(target=worker)
    
    energy.append(energy_2D(S))
    magnetization.append(np.sum(S))
    
#    if animation:
#        t.start()
    S = np.array(S.reshape(S.size))
    print("Running...")
    for n in range(nsteps):
#        S, dE, dM = ising_step_2D(S, beta, H, neighbours)
        p = np.array([np.random.rand() for n in range(S.size)])
        S, dE, dM = ising_step_2D(S, beta, H, neighbours, p)
        energy.append(energy[-1] + dE)
        magnetization.append(magnetization[-1] + dM)
#        if animation and not bool(n%nplot):
#            q.put(S.reshape(shape))
    print("Done running :)")
    S = S.reshape(shape)
    
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

#%%
    
def ising_animation_2D(S, beta, H, 
                       textlabel=lambda i: 'Paso {:.0f}'.format(i)):
    """Makes a series of 2D plots into an animation.
    
    Parameters
    ----------
    S : np.array
        2D spin matrix. Each of its elements should be +1 or -1.
    beta : float
        Multiplicative inverse of the temperature of the system.
    H : float
        External magnetic field. Must be between 1 and -1.
    textlabel : function, optional
        Function that returns a text label for each plot. Must take in only one 
        int parameter and return text.
    
    Returns
    -------
    ising_animation_generator_2D : generator
        Animation generator.    
        
    """
    
    # ACTIVE CODE
    
    fig = plt.figure()
    fig.add_axes()
    ax = fig.gca()
    
    ax.set_xlim((0, S.shape[0]))
    ax.set_ylim((0, S.shape[1]))
    plt.xlabel("X (u.a.)")
    plt.ylabel("Y (u.a.)")
    plt.title(r"Spines con $\beta$={:.2f}, $H$={:.2f}".format(beta, H))
    
    label = ax.text(0.02, 0.93, '', transform=ax.transAxes, 
                    color='r', fontsize='x-large', fontweight='bold')

    def ising_animation_generator_2D():
        i = 0
        while True:
            S = yield
            label.set_text('')
            ax.pcolormesh(S)
            label.set_text(textlabel(i))
            plt.show()
            i = i + 1
            yield
    
    generator = ising_animation_generator_2D()
    generator.send(None)
    generator.send(S)
    next(generator)
    
    return generator