#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:17:36 2018

@author: gabriel
"""
import numpy as np
import matplotlib.pyplot as plt
from ising import ising_step_2D
from ising import energy_2D
import time


start = time.time()

#Tamaño en x,
Lx = 8
#Tamaño en y,
Ly = 8
H = 0
#beta_critico = log(1+sqrt(2))/2 = 0.44069
beta = 0.441

npasos = 10000   #cantidad total de pasos
#100000 pasos equivale a 195s de espera para una grilla de 15x15

#Condición inicial caliente
Sij = np.where(np.random.rand(Lx, Ly)>0.5, 1, -1)
#Condición incial fría
#Sij = np.ones((Lx, Ly)) 

energia = np.zeros((npasos, 1))
magnet = np.zeros((npasos, 1))

energia[0] = energy_2D(Sij)
magnet[0] = np.sum(Sij)

for n in np.arange(npasos - 1):
    Sij, DE, dM = ising_step_2D(Sij, beta, H)
    energia[n + 1] += energia[n] + DE
    magnet[n + 1] = magnet[n] + dM

x = np.arange(1, npasos + 1, 1)
energia_promedio = np.cumsum(energia)/x
magnet_promedio = np.cumsum(magnet)/x

end = time.time()
print("Tiempo de ejecución: {:.4}s".format(end-start))

"""
plt.figure("Energía promedio", clear=True)
plt.plot(energia_promedio,"o")
plt.ylabel("Energía promedio")
plt.xlabel("Paso")

plt.figure("Magnetización promedio", clear=True)
plt.plot(magnet_promedio,"o")
plt.ylabel("Magnetización promedio")
plt.xlabel("Paso")
"""

plt.figure("Magnetización", clear=True)
plt.plot(magnet/(Lx*Ly),"o", markersize=2)
plt.ylabel("Magnetización (u. a.)")
plt.xlabel("Paso")


#Conclusiones:

#Grilla de 15x15
#Para altas temperaturas npre=5000
#Para temperaturas cercanas a Tc npre=50000
#Para temperaturas bajas npre=10000
