#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:49:32 2018

@author: gabriel
"""

import numpy as np
from ising import immediate_neighbours


def ising_step(S, beta):
    """
    Executes one step in the Markov chain using Metropolis algorithm.
    
    Parameters
    ----------
    S : np.array
        A 2D matrix with N rows (up-down xi index) and M columns 
        (right-left yi index).
        
    beta : float
        The multiplicative inverse of the Temperature of the system.
        
    Returns
    -------
    S : np.array
        A 2D matrix with N rows (up-down xi index) and M columns 
        (right-left yi index).
        
    dE : np.array
        Energy change accumulated.
        
    dM : np.array
        Magnetization change accumulated.
    """
    
    Lx, Ly = np.shape(S)
    dE = 0  
    dM = 0 
    
    #For each place in the matrix S, a random spin flip will be proposed. 
    for i in np.arange(Lx):
        for j in np.arange(Ly):
            
            spin_old = S[i, j]
            spin_new = -S[i, j]     #spin flip
            
            right, up, left, down = immediate_neighbours(S, i, j)
            
            #Partial energy difference
            dE_partial = 2*spin_old * S[i,j] * (up + right + left + down)
            
            if dE<0:
                #If energy decrease, spin flip will be accepted.
                S[i, j] = spin_new
                dE = dE + dE_partial
                dM = dM + (spin_new - spin_old)
            else:
                #If energy increase, the change will be drawn.
                p = np.random.rand()
                expbetaE = np.exp(-beta*dE)
                
                if p<expbetaE:
                    #Change will be accepted.
                    S[i, j] = spin_new
                    dE = dE + dE_partial
                    dM = dM + (spin_new - spin_old)
                    
    return S, dE, dM
                    
                
                