# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 16:31:00 2018

@author: Gabriel
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


d=np.load('250Temperaturas_campo0.npz')
e= dict(zip(('beta_vector','energy_matrix','magnetization_matrix','H','npre','nsteps', "enlapsed", "N"),(d[k] for k in d)))

beta_vector = e["beta_vector"]
energy_matrix = e["energy_matrix"]
magnetization_matrix = e["magnetization_matrix"]
N = e["N"]

mean_energy = np.zeros_like(beta_vector)
mean_magnetization = np.zeros_like(beta_vector)
var_energy = np.zeros_like(beta_vector)
var_magnetization = np.zeros_like(beta_vector)

for i in range(len(beta_vector)):
    mean_energy[i] = np.mean(energy_matrix[i, :])
    mean_magnetization[i] = np.mean(magnetization_matrix[i, :])
    var_energy[i] = np.var(energy_matrix[i, :])
    var_magnetization[i] = np.var(magnetization_matrix[i, :])

#heat capacity
C = 1/(N) * var_energy*beta_vector**2

#magnetic susceptibility
chi = 1/(N) *  var_magnetization*beta_vector
#%%

plt.figure()
plt.plot(1/beta_vector, mean_energy/N, "o", markersize=3)
plt.xlabel("Temperatura (k/J)")
plt.ylabel("Energía media")

plt.figure()
plt.plot(1/beta_vector, mean_magnetization/N, "o", markersize=3)
plt.xlabel("Temperatura (j/J)")
plt.ylabel("Magnetización media")

plt.figure()
plt.plot(1/beta_vector, chi, "o", markersize=3)
plt.xlabel("Temperatura (k/J)")
plt.ylabel("Susceptibilidad magnética")

plt.figure()
plt.plot(1/beta_vector, C, "o", markersize=3)
plt.xlabel("Temperatura (k/J)")
plt.ylabel("Capacidad calorífica")


#%%

"""
where = np.where(1/beta_vector)
x_data = 1/beta_vector[where]
y_data = mean_magnetization[where]/N

def critical_exponent(T, Tc, beta_critical_exp, A):
    return A*(Tc - T)**beta_critical_exp

parametros_iniciales = [2.2, 1/8, 1]

plt.figure()
plt.plot(x_data, y_data, "o")
plt.plot(x_data, critical_exponent(x_data, *parametros_iniciales))


popt, pcov = curve_fit(critical_exponent, x_data, y_data, p0=parametros_iniciales)

"""