#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:24:05 2018

@author: gabriel
"""

#script general para hacer una corrida  a un set de parámetros,

import numpy as np
#import matplotlib.axes
import matplotlib.pyplot as plt
from ising2Dpaso import ising2Dpaso
from En import En
import time

start = time.time()


#Tamaño en x,
Lx = 15
#Tamaño en y,
Ly = 15
#beta_critico = log(1+sqrt(2))/2 = 0.44069
beta = 0.5

npre = 2000      #cantidad de pasos hasta termalizar
nshow = 100     #cada cuantos pasos muestra el gráfico
npasos = 5000   #cantidad total de pasos

#Sij es una matriz de 1 y -1 indicando las dos proyecciones de spin

Sij = np.where(np.random.rand(Lx, Ly)>0.5, 1, -1)

"""
fig = plt.figure(1)
fig.add_axes()
ax = fig.gca()
ax.pcolormesh(Sij)
plt.title("Estado inicial")
"""

energia = np.zeros((npasos + 1, 1))
magnet = np.zeros((npasos + 1, 1))

#pretermalizo, a y b son cualquier cosa
for n in np.arange(npre):
    Sij, a, b = ising2Dpaso(Sij, beta)

energia[0] = En(Sij)
magnet[0] = np.sum(Sij)

for n in np.arange(npasos):
    Sij, DE, dM = ising2Dpaso(Sij, beta)
    energia[n + 1] += energia[n] + DE
    magnet[n + 1] = magnet[n] + dM


"""    
    if n%nshow==0:
        ax.pcolormesh(Sij)
        plt.title("T = {}".format(1/beta))      #llos graficos no funcionan
"""

plt.figure(2, clear=True)
plt.subplot(2,1,1)
plt.plot(energia, "o-")
plt.ylabel("Energía")

plt.subplot(2,1,2)
plt.plot(magnet, "o-")
plt.ylabel("Magnetización")
plt.xlabel("Paso")

end = time.time()
print("Tiempo de ejecución: {:.4}s".format(end-start))