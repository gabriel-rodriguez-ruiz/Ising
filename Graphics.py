# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 16:31:00 2018

@author: Gabriel, Valeria
"""

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%% Temperature sweep data

# Load all data into dictionary
d = np.load('250Temperaturas_campo0.npz')
e = dict(zip(('beta_vector','energy_matrix','magnetization_matrix','H',
              'npre','nsteps', "enlapsed", "N"), (d[k] for k in d)))

# Select data
beta_vector = e["beta_vector"]
energy_matrix = e["energy_matrix"]
magnetization_matrix = e["magnetization_matrix"]
#N = e["N"]
N = 32

# Mean parameters
mean_energy = np.zeros_like(beta_vector)
mean_magnetization = np.zeros_like(beta_vector)
var_energy = np.zeros_like(beta_vector)
var_magnetization = np.zeros_like(beta_vector)

for i in range(len(beta_vector)):
    mean_energy[i] = np.mean(energy_matrix[i, :])
    mean_magnetization[i] = np.mean(magnetization_matrix[i, :])
    var_energy[i] = np.var(energy_matrix[i, :])
    var_magnetization[i] = np.var(magnetization_matrix[i, :])

# Heat capacity
C = 1/(N) * var_energy*beta_vector**2

# Magnetic susceptibility
chi = 1/(N) *  var_magnetization*beta_vector

#%% Temperature sweep graphics

rcParams.update({'font.size': 14})
rcParams.update({'lines.linewidth': 3})
rcParams.update({'lines.markersize': 6})

fig = plt.figure()
plt.plot(1/beta_vector, mean_energy/N, "o", markersize=3)
plt.xlabel(r"Temperatura $(J/k_B)$")
plt.ylabel("Energía media (J)")
box = fig.axes[0].get_position()
fig.axes[0].set_position([1.1*box.x0, 1.1*box.y0, .95*box.width, box.height])
plt.savefig('U.pdf', bbox_inches='tight')

fig = plt.figure()
plt.plot(1/beta_vector, mean_magnetization/N, "o", markersize=3)
plt.xlabel(r"Temperatura $(J/k_B)$")
plt.ylabel(r"Magnetización media $(\mu_B)$")
box = fig.axes[0].get_position()
fig.axes[0].set_position([1.1*box.x0, 1.1*box.y0, .95*box.width, box.height])
plt.savefig('M.pdf', bbox_inches='tight')

fig = plt.figure()
plt.plot(1/beta_vector, chi, "o", markersize=3)
plt.xlabel(r"Temperatura $(J/k_B)$")
plt.ylabel(r"Susceptibilidad magnética")
box = fig.axes[0].get_position()
fig.axes[0].set_position([1.35*box.x0, 1.1*box.y0, .95*box.width, box.height])
plt.savefig('Chi.pdf', bbox_inches='tight')

fig = plt.figure()
plt.plot(1/beta_vector, C, "o", markersize=3)
plt.xlabel(r"Temperatura $(J/k_B)$")
plt.ylabel(r"Capacidad calorífica $(k_B)$")
box = fig.axes[0].get_position()
fig.axes[0].set_position([1.35*box.x0, 1.1*box.y0, .95*box.width, box.height])
plt.savefig('C.pdf', bbox_inches='tight')

#%% Critical Temperature

# Where is the peak?
where = np.where(C > 1.6)

# Assume critical temperature as mean temperature of the peak zone
Tc = np.mean(1/beta_vector[where])
Tc_error = np.std(1/beta_vector[where])/np.sqrt(len(where))

#%% Critical Exponent Nonlinear Fit

# First find the area that should be fit
where = np.where(
        (1/beta_vector <= Tc) & (1/beta_vector > 2) & (mean_magnetization > 0)) # 
x_data = 1/beta_vector[where] # temperature
y_data = mean_magnetization[where]/N # magnetization

# Plot the data to fit
fig = plt.figure()
plt.plot(1/beta_vector, mean_magnetization/N, "o")
plt.plot(x_data, y_data, "o")
#plt.plot(1/beta_vector, mean_magnetization/N, "o", markersize=3)
#plt.plot(x_data, critical_exponent(x_data, *initial_parameters))
plt.xlabel(r"Temperatura $(J/k_B)$")
plt.ylabel(r"Magnetización media $(\mu_B)$")

# Then define the function to fit
def critical_exponent(T, beta_critical_exp, A, a):
    return A*((a - T)/a)**beta_critical_exp
initial_parameters = [1, 1/8, 4.6] # 1 1/8 2.2932

# Now fit the data
popt, pcov = curve_fit(critical_exponent, x_data, y_data, 
                       p0=initial_parameters)
std = np.diag(pcov)
beta_critical_exp = popt[0]

# Then add the model to the plot
x = np.append(np.linspace(2, 3, 100000), 2.2932262072855543)
plt.plot(x, critical_exponent(x, *popt), label="Modelo")
plt.legend()

# And now can also plot Onsager's model
#x = np.linspace(1, 2.27, 1000)
#plt.plot(x, 1*(2.27 - x)**(1/8), label="Onsager")
fig.axes[0].set_position([1.35*box.x0, 1.1*box.y0, .95*box.width, box.height])
plt.savefig('Modelo.pdf', bbox_inches='tight')
