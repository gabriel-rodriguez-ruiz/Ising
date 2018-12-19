#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:30:44 2018

@author: Gabriel, Valeria
"""

import numpy as np
from math import exp
from matplotlib import rcParams
import matplotlib.pyplot as plt
import os
import queue
import time
import threading

#%%%%%%%%%%%%%%%%%%%%%%%%% MAIN FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

def initial_condition_2D(condition, shape):
    """Returns the initial spin matrix S depending on condition.
    
    Parameters
    ----------
    condition : str
        Initial conditions' descriptor. Must be either 'hot' (random 
        initial condition) or 'cold' (uniform initial condition full of 
        parallel spins).
    shape : tuple
        Spin matrix dimensions. Each of its elements should be an int: 
        (N, M) where N is the amount of rows (x-axis) and M is the 
        amount of columns (y-axis).

    """
    
    if condition == "hot": # Random spins
        S = np.where(np.random.rand(*shape) > 0.5, 1, -1)        
    elif condition == "cold":
        S = np.ones(shape) # Alligned spins
    else:
        raise TypeError("Wrong condition. Must be 'hot' or 'cold'")
    
    return S

#%%

def energy_2D(S):
    """Returns energy from a spin matrix S.
    
    Parameters
    ----------
    S : np.array
        A 2D spin matrix. Each of its element must be either +1 or -1.
    
    Returns
    -------
    En : float
        Energy of that specific spin configuration.
    
    See Also
    --------
    all_immediate_neighbours_2D
    
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
    
    See Also
    --------
    ising_simulation_2D
    
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
            expbetaE = exp(-beta * dE_partial)
            # It will only be accepted with probability exp(-beta*dE_partial)
            if pk < expbetaE:
                S[k] = -Sij
                dE = dE + dE_partial
                dM = dM - 2*Sij
        
        k += 1
                    
    return S, dE, dM

#%%

def ising_simulation_2D(S, beta, H=0, nsteps=1000, printing=True, 
                        q=None, results=None, nplots=4):
    """Executes several steps in a Markov chain using Metropolis algorithm.

    This function, when used on its own, returns final spin matrix, energy and 
    magnetization. When run by the Ising animation function that fills some 
    optional parameters, it doesn't return any data.
    
    Parameters
    ----------
    S : np.array
        2D spin matrix. Each of its elements should be +1 or -1.
    beta : float
        Multiplicative inverse of the temperature of the system.
    H=0 : float, optional
        External magnetic field. Must be between 1 and -1.
    nsteps=1000 : int, optional
        Desired amount of steps.
    q=None : queue.Queue, optional
        Queue related to a thread that plots on animation.
    results=None : dict, optional
        Dictionary where animation saves final data.
    nplots=4 : int, optional
        Number of plots the animation should have.
        
    Returns
    -------
    Sf : np.array
        Final 2D spin matrix. Each of its elements are +1 or -1.
    E : np.array
        Accumulated energy change array. Holds one value per step.
    M : np.array
        Accumulated magnetization change array. Holds one value per step.
    
    See Also
    --------
    energy_2D
    ising_step_2D
    ising_animation_2D
    all_immediate_neighbours_2D
    flat_matrix_index_conversion_2D    
    
    """

    animation = q is not None
    nstepsbetween = int(nsteps/nplots)
    
    shape = S.shape
    neighbours = all_immediate_neighbours_2D(shape)
    matrix_to_flat = flat_matrix_index_conversion_2D(shape)[1]
    neighbours = [tuple(matrix_to_flat(index) for index in nij) 
                  for nij in neighbours]

    energy = []
    magnetization = []       
    
    energy.append(energy_2D(S))
    magnetization.append(np.sum(S))
    
    S = np.array(S.reshape(S.size))
    if printing:
        print("Running...")
        start = time.time()
    for n in range(nsteps):
        p = np.array([np.random.rand() for n in range(S.size)])
        S, dE, dM = ising_step_2D(S, beta, H, neighbours, p)
        energy.append(energy[-1] + dE)
        magnetization.append(magnetization[-1] + dM)
        if animation:
            if not bool((n+1) % nstepsbetween):
                data = [np.array(S), np.array(energy), np.array(magnetization)]
                q.put(data)
    if printing:
        end = time.time()
        print("Done running :)")
        print("Enlapsed: {:.2f} s".format(end-start))
    S = S.reshape(shape)
    
    energy = np.array(energy)
    magnetization = np.array(magnetization)

    if animation:
        results.update(dict(S=S,
                            energy=energy,
                            magnetization=magnetization))
    else:
        return S, energy, magnetization

#%%

def ising_animation_2D(S, beta, H=0, nsteps=1000, nplots=4, 
                       full=False, printing=True, save=False):
    """Executes and plots many steps in a Markov chain by Metropolis algorithm.
    
    Beware! This code uses runs on two thrads that are independent from the 
    main thread. Because of that, the simulation is non-blocking but the 
    results will only appear once it's over.
    
    Parameters
    ----------
    S : np.array
        2D spin matrix. Each of its elements should be +1 or -1.
    beta : float
        Multiplicative inverse of the temperature of the system.
    H=0 : float, optional
        External magnetic field. Must be between 1 and -1.
    nsteps=1000 : int, optional
        Desired amount of steps.
    nplots=4 : int, optional
        Number of plots the animation should have.
    full=False : bool
        Parameter that allows full animation, which additionally includes 
        medium energy and magnetization.
    printing=True : bool
        Parameter that decides whether to print some messages or not.
    save=False : bool
        Parameter that allows to save each picture shown in order to later make 
        a video or gif.
    
    Returns
    -------
    results : dict
        Results of the simulation, which will only be filled once the 
        simulation is over.
            Sf : np.array
                Final 2D spin matrix. Each of its elements should be +1 or -1.
            energy : np.array
                Energy as a function of the number of steps.
            magnetization : np.array
                Magnetization as a function of the number of steps.
    
    See Also
    --------
    ising_partial_animation_2D
    ising_full_animation_2D
    
    """
    
    # Plot configuration
    rcParams.update({'font.size': 14})
    rcParams.update({'lines.linewidth': 3})
    
    # Save parameters
    if save:
        folder = 'Beta={} H={} '.format(beta, H)
        folder = folder + '{}x{}'.format(S.shape[0], S.shape[1])
        folder = os.path.join(os.getcwd(), 'Video', folder)
        folder = new_dir(folder, newformat='{} ({})')
        filename = lambda i : os.path.join(folder, '{:.0f}.png'.format(i))
    else:
        filename = None
    
    # Initialize animation
    if full:
        results, t, q = ising_full_animation_2D(S, beta, H, nsteps, 
                                                nplots, save, filename)
    else:
        results, t, q = ising_partial_animation_2D(S, beta, H, nsteps, 
                                                   nplots, save, filename)
    
    # Start simulation
    t.start()
    def ising():
        ising_simulation_2D(S, beta, H=H, nsteps=nsteps, q=q, results=results, 
                            nplots=nplots, printing=printing)
        q.put(False)
    t2 = threading.Thread(target=ising)
    time.sleep(1)
    if printing:
        print("Beware! You must wait for the simulation to end.")
    t2.start()
    
    return results
    
#%%%%%%%%%%%%%%%%%%%%%%% ANIMATION TOOLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%#

def ising_partial_animation_2D(S, beta, H, nsteps, nplots, save, filename):
    """Light animation which plots steps done by Metropolis algorithm.
    
    Beware! This code uses runs on two thrads that are independent from the 
    main thread. Because of that, the simulation is non-blocking but the 
    results will only appear once it's over.
    
    Parameters
    ----------
    S : np.array
        2D spin matrix. Each of its elements should be +1 or -1.
    beta : float
        Multiplicative inverse of the temperature of the system.
    H=0 : float, optional
        External magnetic field. Must be between 1 and -1.
    nsteps=1000 : int, optional
        Desired amount of steps.
    nplots=4 : int, optional
        Number of plots the animation should have.
    full=False : bool
        Parameter that allows full animation, which additionally includes 
        medium energy and magnetization.
    save=False : bool
        Parameter that allows to save each picture shown in order to later make 
        a video or gif.
    filename : function
        Function that designes filenames according to frame number.
    
    Returns
    -------
    results : dict
        Results of the simulation, which will only be filled once the 
        simulation is over.
            Sf : np.array
                Final 2D spin matrix. Each of its elements should be +1 or -1.
            energy : np.array
                Energy as a function of the number of steps.
            magnetization : np.array
                Magnetization as a function of the number of steps.
    t : threading.Thread
        Thread that plots
    q : queue.Queue
        Queue that holds data to be plotted.
    fig : plt.Figure
        Matplotlib figure where data will be plotted.
    
    See Also
    --------
    ising_animation_2D
    threading.Thread
    queue.Queue
    
    """
    
    # General configuration
    S0 = S
    shape = S0.shape
    nstepsbetween = int(nsteps/nplots)
    
    # Figure configuration
    fig = plt.figure()
    fig.add_axes()
    ax = fig.gca()
    
    # Plot configuration --> Spin matrix
    plt.title(r"Spines con $T$={:.1f} ".format(1/beta) + 
              r"$J\,/\,k_B$, " + r"$H$={} $\mu_B$".format(H))
    plt.xlabel("X (u.a.)")
    plt.ylabel("Y (u.a.)")
    ax.set_xlim((0, S.shape[0]))
    ax.set_ylim((0, S.shape[1]))
    label = ax.text(0.02, 0.93, '', transform=ax.transAxes, 
                    color='r', fontsize='x-large', fontweight='bold')

    # Generator that will plot every time some data is passed to it
    def ising_animation_generator_2D():
        i = 0
        while True:
            S = yield # This is how you pass data to it
            S = S.reshape(shape)
            ax.pcolormesh(S.T)
            plt.show()
            if i != nplots + 1:
                label.set_text("Paso: {:.0f}".format(i*nstepsbetween))
                if save and i != 0:
                    plt.savefig(filename(i), bbox_inches='tight')
            else:
                time.sleep(2)
                label.set_text("Paso: {:.0f}".format(0))
                if save and i != 0:
                    plt.savefig(filename(0), bbox_inches='tight')
            i = i + 1
            yield # Here it finishes a round and waits for the next one
    
    # Initialization of the generator
    generator = ising_animation_generator_2D()
    generator.send(None)
    generator.send(S)
    next(generator)
    
    # Dictionary where the results will be saved
    results = {'Sf': S, 'energy': np.zeros(nsteps+1), 
               'magnetization': np.zeros(nsteps+1)}
    
    # Thread that passes data to the generator
    q = queue.Queue()
    fig = plt.gcf()
    def plot():
        plt.ion()
        time.sleep(2)
        while True:
            S = q.get() # Waits for data to be available
            if isinstance(S, bool):
                generator.send(S0)
                next(generator)
                fig.canvas.draw()
                time.sleep(.5)
                break
            generator.send(S[0])
            next(generator)
            fig.canvas.draw()
            time.sleep(.5)
    t = threading.Thread(target=plot)
    
    return results, t, q

#%%
    
def ising_full_animation_2D(S, beta, H, nsteps, nplots, save, filename):
    """Light animation which plots steps done by Metropolis algorithm.
    
    Beware! This code uses runs on two thrads that are independent from the 
    main thread. Because of that, the simulation is non-blocking but the 
    results will only appear once it's over.
    
    Parameters
    ----------
    S : np.array
        2D spin matrix. Each of its elements should be +1 or -1.
    beta : float
        Multiplicative inverse of the temperature of the system.
    H=0 : float, optional
        External magnetic field. Must be between 1 and -1.
    nsteps=1000 : int, optional
        Desired amount of steps.
    nplots=4 : int, optional
        Number of plots the animation should have.
    save=False : bool
        Parameter that allows to save each picture shown in order to later make 
        a video or gif.
    filename : function
        Function that designes filenames according to frame number.
    
    Returns
    -------
    results : dict
        Results of the simulation, which will only be filled once the 
        simulation is over.
            Sf : np.array
                Final 2D spin matrix. Each of its elements should be +1 or -1.
            energy : np.array
                Energy as a function of the number of steps.
            magnetization : np.array
                Magnetization as a function of the number of steps.
    t : threading.Thread
        Thread that plots
    q : queue.Queue
        Queue that holds data to be plotted.
    fig : plt.Figure
        Matplotlib figure where data will be plotted.
    
    See Also
    --------
    ising_animation_2D
    threading.Thread
    queue.Queue
    
    """
    
    # General configuration
    data0 = [S, np.array([energy_2D(S)]), np.array([np.sum(S)])]
    shape = S.shape
    nstepsbetween = int(nsteps/nplots)
    
    # Figure configuration
    fig = plt.figure()
    plt.suptitle(r"Spines con $T$={:.1f} ".format(1/beta) + 
                 r"$J\,/\,k_B$, " + r"$H$={} $\mu_B$".format(H))
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    grid = plt.GridSpec(2, 4, wspace=0.7, hspace=0.2)
    ax1 = plt.subplot(grid[:,:2])
    ax2 = plt.subplot(grid[0,2:])
    ax3 = plt.subplot(grid[1,2:])

    # First plot configuration --> Spin matrix
    ax1.set(adjustable='box-forced', aspect='equal') 
    ax1.set_xlim((0, S.shape[0]))
    ax1.set_ylim((0, S.shape[1]))
    ax1.set_xlabel("X (u.a.)")
    ax1.set_ylabel("Y (u.a.)")
    label = ax1.text(0.02, 0.93, '', transform=ax1.transAxes, 
                    color='r', fontsize='x-large', fontweight='bold')
    
    # Second plot configuration --> Medium Energy
    ax2.set_xlim((0,nsteps))
    ax2.set_ylabel(r"Energía media ($J$)")
    
    # Third plot configuration --> Medium Magnetization
    ax3.set_xlim((0,nsteps))
    ax3.set_ylabel(r"Magnetización media ($\mu_B$)")
    ax3.set_xlabel("Paso (u.a.)")

    # Generator that will plot every time some data is passed to it
    def ising_animation_generator_2D():
        i = 0
        while True:
            data = yield
            S, E, M = data
            S = S.reshape(shape)
            n = np.arange(1, len(E)+1)
            E = np.cumsum(np.array(E))/n
            M = np.cumsum(np.array(M))/n
            ax1.pcolormesh(S.T)
            ax2.plot(range(len(E)), E, 'b')
            ax3.plot(range(len(M)), M, 'b')
            if i != nplots + 1:
                label.set_text("Paso: {:.0f}".format(i*nstepsbetween))
                if save and i != 0:
                    plt.savefig(filename(i), bbox_inches='tight')
            else:
                label.set_text("Paso: {:.0f}".format(0))
                if save and i != 0:
                    plt.savefig(filename(0), bbox_inches='tight')
            i = i + 1
            yield
    
    # Initialization of the generator
    generator = ising_animation_generator_2D()
    generator.send(None)
    generator.send(data0)
    next(generator)
    
    # Dictionary where the results will be saved
    results = {'Sf': S, 'energy': np.zeros(nsteps+1), 
               'magnetization': np.zeros(nsteps+1)}

    # Thread that passes data to the generator
    q = queue.Queue()
    fig = plt.gcf()
    def plot():
        plt.ion()
        time.sleep(2)
        while True:
            data = q.get() # Waits for data to be available
            if isinstance(data, bool):
                generator.send(data0)
                time.sleep(1)
                next(generator)
                fig.canvas.draw()
                break
            generator.send(data)
            time.sleep(1)
            next(generator)
            fig.canvas.draw()
    t = threading.Thread(target=plot)
    
    return results, t, q

#%%%%%%%%%%%%%%%%%%%%%%%%% UTILITIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    
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
    
    See Also
    --------
    immediate_neighbours_2D
    
    """
    
    N, M = shape
    neighbours = []
    
    for i in range(N):
        for j in range(M):
            neighbours.append(immediate_neighbours_2D(shape, (i, j)))
        
    return neighbours

#%%

def new_dir(my_dir, newformat='{}_{}'):
    
    """Makes and returns a new directory to avoid overwriting.
    
    Takes a directory name 'my_dir' and checks whether it already 
    exists. If it doesn't, it returns 'dirname'. If it does, it 
    returns a related unoccupied directory name. In both cases, 
    the returned directory is initialized.
    
    Parameters
    ----------
    my_dir : str
        Desired directory (should also contain full path).
    
    Returns
    -------
    new_dir : str
        New directory (contains full path)
    
    Yields
    ------
    new_dir : directory
    
    """
    
    sepformat = newformat.split('{}')
    base = os.path.split(my_dir)[0]
    
    new_dir = my_dir
    while os.path.isdir(new_dir):
        new_dir = os.path.basename(new_dir)
        new_dir = new_dir.split(sepformat[-2])[-1]
        try:
            new_dir = new_dir.split(sepformat[-1])[0]
        except ValueError:
            new_dir = new_dir
        try:
            new_dir = newformat.format(my_dir, str(int(new_dir)+1))
        except ValueError:
            new_dir = newformat.format(my_dir, 2)
        new_dir = os.path.join(base, new_dir)
    os.makedirs(new_dir)
        
    return new_dir