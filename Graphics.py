# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 16:31:00 2018

@author: Gabriel
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

d=np.load('400Temperaturas.npz')
e= dict(zip(('beta_vector','energy_matrix','magnetization_matrix','H','npre','nsteps', "enlapsed", "N"),(d[k] for k in d)))

beta_vector = e["beta_vector"]
energy_matrix = e["energy_matrix"]
magnetization_matrix = e["magnetization_matrix"]
#N = e["N"]
N = 32

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
plt.xlabel(r"Temperatura $(J/k_B)$")
plt.ylabel("Energía media (J)")

plt.figure()
plt.plot(1/beta_vector, mean_magnetization/N, "o", markersize=3)
plt.xlabel(r"Temperatura $(J/k_B)$")
plt.ylabel(r"Magnetización media $(\mu_B)$")

plt.figure()
plt.plot(1/beta_vector, chi, "o", markersize=3)
plt.xlabel(r"Temperatura $(J/k_B)$")
plt.ylabel(r"Susceptibilidad magnética")

plt.figure()
plt.plot(1/beta_vector, C, "o", markersize=3)
plt.xlabel(r"Temperatura $(J/k_B)$")
plt.ylabel(r"Capacidad calorífica $(k_B)$")


#%%   Critical Temperature

where = np.where(C>1.6)

Tc = np.mean(1/beta_vector[where])
Tc_error = np.std(1/beta_vector[where])/np.sqrt(len(where))


#%%

where = np.where((1/beta_vector<=Tc) & (1/beta_vector>2) & (mean_magnetization>0))
x_data = 1/beta_vector[where]
y_data = mean_magnetization[where]/N

def critical_exponent(T, beta_critical_exp, A):
    return A*((2.2932 - T)/2.2932)**beta_critical_exp

parametros_iniciales = [1,1/8]

plt.figure()
plt.plot(1/beta_vector, mean_magnetization/N, "o")
plt.plot(x_data, y_data, "o")
#plt.plot(1/beta_vector, mean_magnetization/N, "o", markersize=3)

#plt.plot(x_data, critical_exponent(x_data, *parametros_iniciales))

x = np.append(np.linspace(2, 3, 100000), 2.2932262072855543)

popt, pcov = curve_fit(critical_exponent, x_data, y_data, p0=parametros_iniciales)
std = np.diag(pcov)
plt.plot(x, critical_exponent(x, *popt), label="Modelo")

#x = np.linspace(1, 2.27, 1000)
#plt.plot(x, 1*(2.27 - x)**(1/8), label="Onsager")

plt.xlabel(r"Temperatura $(J/k_B)$")
plt.ylabel(r"Magnetización media $(\mu_B)$")

beta_critical_exp = popt[0]
plt.legend()