#!/usr/bin/env python3
"""
Calculate energy curves in the Fermi-Hubbard model over different filling
fractions, along with the corresponding order parameter assuming either a
ferromagnetic or antiferromagnetic Ansatz. Also computes the optimum energy
and order parameter for a fixed occupation.

Created March 19, 2019
by Johann Gan
"""

import numpy as np

def calc_magnetization(densityup, densitydown):
    """
    Calculate magnetization for a given density of spin-up and spin-down
    
    Inputs:
        densityup(down) = density (per site) of spin-up(down) particles
    Outputs:
        magnetization
    """
    if densityup == 0 and densitydown == 0: # no particles --> no magnetization
        return 0
    return (densityup - densitydown) / (densityup + densitydown)

def calc_energy_curve_fm(nsites, nparticles, U, t):
    """
    Calculate energy of system assuming ferromagnetic order, for each possible
    magnetization value under given parameters. At equilibrium, energy would be
    minimized

    Inputs:
        nsites = number of sites
        nparticles = total number of particles in the system
        U, t = Hubbard parameters
    Outputs:
        magnetizations, energies = magnetizations and corresponding energies
    """
    # Go through each possibility of number of up/down spins
    magnetizations = np.array([])
    energies = np.array([])
    # Symmetric under exchange of up/down, so only need distinct pairs
    for ndown in range(int(nparticles//2 + 1)):
        nup = int(nparticles - ndown)
        # Overfilled
        if nup > nsites or ndown > nsites:
            continue
        densityup = nup / nsites
        densitydown = ndown / nsites

        # Fill up the lowest-energy momentum states
        allowed_momenta = np.linspace(-np.pi + 2*np.pi/nsites, np.pi, nsites)
        # kinetic energy in ascending order
        allowed_kinetic_energies = np.sort(-2*t*np.cos(allowed_momenta))
        # Analytic formula for energy
        energy = np.sum(allowed_kinetic_energies[:nup]) \
            + np.sum(allowed_kinetic_energies[:ndown]) \
            + 2*U * nup*ndown / nsites
        energy /= nsites    # Normalize to number of sites
        energy -= U*densityup*densitydown   # Add potential part
        magnetizations = np.append(magnetizations,
            calc_magnetization(densityup, densitydown))
        energies = np.append(energies, energy)
    return magnetizations, energies

def calc_min_energy_fm(nsites, nparticles, U, t):
    """
    Calculate the minimum (equilibrium) energy and corresponding magnetization
    for a given system assuming ferromagnetic order
    
    Inputs:
        nsites = number of sites
        nparticles = total number of particles in the system
        U, t = Hubbard parameters
    Outputs:
        magnetization, energy = magnetization (magnitude) and corresponding 
            energy
    """
    magnetizations, energies = calc_energy_curve_fm(nsites, nparticles, U, t)
    min_i = np.argmin(energies)
    return np.abs(magnetizations[min_i]), energies[min_i]

def calc_energy_curve_afm(nsites, nparticles, U, t):
    """
    Calculate energy of system assuming antiferromagnetic order, for each
    possible staggered magnetization value under given parameters. At
    equilibrium, energy would be minimized

    Inputs:
        nsites = number of sites
        nparticles = total number of particles in the system
        U, t = Hubbard parameters
    Outputs:
        staggered_magnetizations, energies = staggered magnetizations and
            corresponding energies
    """
    # Must be an even number of sites and particles
    if nsites % 2 != 0 or nparticles % 2 != 0:
        raise ValueError("Must have even number of sites and particles.")
    elif nparticles > 2*nsites: # Overfilled
        return None, None
    # Go through each possible staggered magnetization
    resolution = 1e-2
    staggered_magnetizations = np.linspace(0, 1, np.ceil(1/resolution))
    energies = np.array([])
    for m in staggered_magnetizations:
        # Fill up the lowest-energy momentum states
        allowed_momenta = np.linspace(-np.pi/2 + 2*np.pi/nsites, np.pi/2, nsites/2)
        # Evenly distributed between spin-up and spin-down
        density = nparticles / (2*nsites)
        # Staggered magnetization interaction energy shift
        delta = m * density * U
        # Analytic formula for allowed energies in ascending order
        allowed_energies = np.sqrt((2*t*np.cos(allowed_momenta))**2 + delta**2)
        # Plus-or-minus
        allowed_energies = np.append(allowed_energies, -allowed_energies)
        # Add interaction with m = 0 unstaggered field
        allowed_energies += U * density
        # Ascending order
        allowed_energies = np.sort(allowed_energies)

        # Populate lowest energy levels
        energy = 2*np.sum(allowed_energies[:int(nparticles//2)])  # For spin-up/down
        energy /= nsites    # Normalize to number of sites
        energy -= U*(1 - m**2)*density**2  # Add field-field interaction part
        energies = np.append(energies, energy)
    return staggered_magnetizations, energies

def calc_min_energy_afm(nsites, nparticles, U, t):
    """
    Calculate the minimum (equilibrium) energy and corresponding staggered
    magnetization for a given system assuming antiferromagnetic order
    
    Inputs:
        nsites = number of sites
        nparticles = total number of particles in the system
        U, t = Hubbard parameters
    Outputs:
        staggered_magnetization, energy = staggered magnetization and
            corresponding energy
    """
    staggered_magnetizations, energies = calc_energy_curve_afm(nsites,
        nparticles, U, t)
    min_i = np.argmin(energies)
    return staggered_magnetizations[min_i], energies[min_i]
