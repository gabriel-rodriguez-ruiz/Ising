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

def immediate_neighbours(S, i, j):
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
    >> 
    """
    
    N, M = np.shape(S)
    
    # Find immediate neighbours up and down
    if i != N - 1:
        inext = i + 1
    else: # Periodic contour conditions (N+1 = 0)
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
    else: # Periodic contour conditions (N+1 = 0)
        jnext = 0
    if i != 0:
        jprevious = j - 1
    else:
        jprevious = M - 1
    right = S[i, jnext]
    left = S[i, jprevious]
    
    return right, up, left, down

def energy(S):
    """
    Returns energy from a spin matrix S.
    
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
            right, up, left, down = immediate_neighbours(S, i, j)
            
            # Now make an addition to energy
            En = En - S[i,j] * (up + right + left + down)

    # Since each pair of spins was counted twice...
    En = En/2

    return En


def ising2Dpaso(Sij, beta):
    """
    Función que devuelve Sij, Dec, dM.
    """
    
    Lx, Ly = np.shape(Sij)
    DEc = 0 #diferencia acumulada desde la ejecución
    dM = 0 #diferencia de magnetización acumulada
    
    #Recorro cada uno de los sitios para proponer un cambio aleatorio
    for i in np.arange(Lx):
        for j in np.arange(Ly):
            
            spin_old = Sij[i, j]
            #propongo invertir ese spin y ver cómo cambia la energía
            spin_new = -Sij[i, j]
            
            #Busco primeros vecinos con condiciones periódicas de contorno

            if i!=(Lx - 1):
                iright = i + 1
            else:
                iright = 0
            
            if i!=0:
                ileft = i - 1
            else:
                ileft = Lx - 1
                
            if j!=(Ly - 1):
                jdown = j + 1
            else:
                jdown = 0
                
            if j!=0:
                jup = j - 1
            else:
                jup = Lx - 1 
            
            #diferencia de energía
            DE = 2*spin_old*(Sij[iright, j] + Sij[ileft, j] + Sij[i, jup] + Sij[i, jdown])
            
            if DE<0:
                #Si la energía disminuye, se acepta el cambio
                Sij[i, j] = spin_new
                DEc = DEc + DE
                dM = dM + (spin_new - spin_old)
            else:
                #Sino, se sortea si se mantiene el cambio
                p = np.random.rand()
                expbetaE = np.exp(-beta*DE)
                
                if p<expbetaE:
                    #Se acepta el cambio
                    Sij[i, j] = spin_new
                    DEc = DEc + DE
                    dM = dM + (spin_new - spin_old)
                    
    return Sij, DEc, dM
                    