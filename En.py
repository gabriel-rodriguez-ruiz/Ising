#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:30:44 2018

@author: gabriel
"""

import numpy as np

def En(Sij):
    """
    Función de energía. Dada una matriz de spines Sij, devuelve la energía.
    """
    Lx, Ly = np.shape(Sij)
    En = 0
    
    for i in np.arange(Lx):
        for j in np.arange(Ly):
            
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
                
            En = En - Sij[i, j]*(Sij[i, jup] + Sij[i, jdown] + Sij[ileft, j] + Sij[iright, j])

    #Como se contó cada par de spines dos veces
    En = En/2

    return En 