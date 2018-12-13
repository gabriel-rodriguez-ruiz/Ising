#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:49:32 2018

@author: gabriel
"""

import numpy as np

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
                    
                
                