#!/usr/bin/env python3
"""
Solves a finite-sized 1D Fermi-Hubbard system with periodic boundaries using
exact diagonalization. Block-diagonalizes each subspace with fixed (nup, ndown)
for improved efficiency, since the Hamiltonian conserves particle number

Created March 19, 2019
by Johann Gan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fclusterdata   # For clustering

def add_fermions(nsites, configurations, latest_added, n_to_add):
    """
    Recursively adds extra fermions one-by-one to form all possible occupations
    with a given number of total fermions and number of sites.

    Inputs:
        nsites = number of sites
        configurations = list of configurations, each of which is a list where
            each site represents the occupation (1) or absence (0) of a fermion
        latest_added = list of the largest occupied site index for each
            configuration (for internal use)
        n_to_add = number of extra fermions to add to existing configurations
    Outputs:
        augmented_configurations = list of configurations with more fermions
    """
    if n_to_add == 0:
        return [[occ for occ in cfg] for cfg in configurations] # Deep copy

    augmented_configurations = []
    augmented_latest_added = []
    for cfg, latest in zip(configurations, latest_added):
        # Add a fermion in each possible spot
        for i in range(latest+1, nsites):
            augmented_configurations.append(list(cfg))
            augmented_configurations[-1][i] = 1
            augmented_latest_added.append(i)
    return add_fermions(nsites, augmented_configurations,
        augmented_latest_added, n_to_add - 1)

def enumerate_basis(nsites, nup, ndown):
    """
    Enumerates a basis of nup spin-up fermions and ndown spin-down fermions
    occupying a chain with nsites sites

    Inputs:
        nsites = number of sites
        nup = number of spin-up fermions
        ndown = number of spin-down fermions
    Outputs:
        up_occupations, down_occupations = list of tuples, where the ith tuple
            represents the occupation (1) or absence (0) of a fermion on each
            site
        basis_idx = dictionary that maps (up_occupation, down_occupation),
            where up_occupation and down_occupation are tuples, to its basis
            index
    """
    up_configurations = add_fermions(nsites, [nsites*[0]], [-1], nup)
    down_configurations = add_fermions(nsites, [nsites*[0]], [-1], ndown)
    i = -1
    basis_size = len(up_configurations)*len(down_configurations)
    up_occupations = basis_size*[None]
    down_occupations = basis_size*[None]
    basis_idx = {}
    # Run through and enumerate all configurations of spin-up+spin-down on the
    # lattice
    for up in up_configurations:
        for down in down_configurations:
            i += 1
            up_tuple = tuple(up)
            down_tuple = tuple(down)
            up_occupations[i] = up_tuple
            down_occupations[i] = down_tuple
            basis_idx[(up_tuple, down_tuple)] = i
    return up_occupations, down_occupations, basis_idx

def build_hamiltonian(nsites, nup, ndown, t, U, mu=0):
    """
    Build the Hamiltonian submatrix (with fixed nup, ndown) in particle-hole
    symmetric form

    Inputs:
        nsites = system size
        nup = number of spin-up particles
        ndown = number of spin-down particles
        t, U = kinetic and potential parameters
        mu (default 0) = chemical potential
    Outputs:
        H = hamiltonian matrix
    """
    # Enumerate the basis
    up_occupations, down_occupations, basis_idx = enumerate_basis(nsites, nup, ndown)
    basis_size = len(up_occupations)
    H = np.asmatrix(np.zeros((basis_size, basis_size))) # Initialize
    for i in range(basis_size):
        n_spin_up = np.array(up_occupations[i])
        n_spin_down = np.array(down_occupations[i])
        # Diagonal
        # Particle-hole symmetric form (1/2 offsets)
        H[i, i] = U * np.sum((n_spin_up-1/2)*(n_spin_down-1/2)) \
            - mu * np.sum(n_spin_up + n_spin_down)
        # Off-diagonal
        # Spin-up moves
        n_spin_tmp = np.copy(n_spin_up)  # work array
        for site, n_site in enumerate(n_spin_up):
            if n_site > 0:  # This site has something that can hop
                for new_site in [(site-1) % n, (site+1) % n]:   # Destination
                    if n_spin_up[new_site] == 0:    # not occupied
                        # Hop
                        n_spin_tmp[site] -= 1
                        n_spin_tmp[new_site] += 1
                        j = basis_idx[(tuple(n_spin_tmp), tuple(n_spin_down))]
                        H[i, j] = -t
                        # Only time sign might be an issue is at the periodic
                        # boundary
                        if np.abs(site - new_site) > 1:
                            # After destroying one, an odd number of order
                            # switches are left, which inverts the sign
                            if np.sum(n_spin_up) % 2 == 0:
                                H[i, j] = t
                        # Reset for the next possibility
                        n_spin_tmp[site] += 1
                        n_spin_tmp[new_site] -= 1
        # Spin-down moves
        n_spin_tmp = np.copy(n_spin_down)
        for site, n_site in enumerate(n_spin_down):
            if n_site > 0:  # This site has something that can hop
                for new_site in [(site-1) % n, (site+1) % n]:   # Destination
                    if n_spin_down[new_site] == 0:    # not occupied
                        # Hop
                        n_spin_tmp[site] -= 1
                        n_spin_tmp[new_site] += 1
                        j = basis_idx[(tuple(n_spin_up), tuple(n_spin_tmp))]
                        H[i, j] = -t
                        # Only time sign might be an issue is at the periodic
                        # boundary
                        if np.abs(site - new_site) > 1:
                            # After destroying one, an odd number of order
                            # switches are left, which inverts the sign
                            if np.sum(n_spin_down) % 2 == 0:
                                H[i, j] = t
                        # Reset for the next possibility
                        n_spin_tmp[site] += 1
                        n_spin_tmp[new_site] -= 1
    return H

def identify_center_band(energies, k):
    """
    Tries to identify the bounds of the central energy band (around the Fermi
    energy) using hierarchical clustering.

    Inputs:
        energies = sorted energy levels
        k = assumed number of distinct bands
    Outputs:
        central_band = energies in the identified central band
    """
    E = np.copy(energies)
    E.resize((E.shape[0], 1))   # Transpose
    labels = fclusterdata(E, k, criterion='maxclust')   # Clustering
    # Find the levels closest to the Fermi level
    return energies[np.where(labels == labels[len(energies)//2])]

# System parameters
n = 6   # number of sites
t = 1   # hopping amplitude
U = 100   # on-site interaction
mu = 0  # chemical potential

energies = np.array([])
# Hamiltonian conserves total particle number, so we can diagonalize each
# subspace with fixed (nup, ndown) separately, and tabulate the energies from
# each
for nup in range(n+1):
    for ndown in range(nup+1):
        H = build_hamiltonian(n, nup, ndown, t, U, mu)
        energy_subset = np.linalg.eigvalsh(H)
        energies = np.append(energies, energy_subset)
        if nup != ndown:
            energies = np.append(energies, energy_subset)
energies.sort()
Efermi = np.median(energies)    # Fermi energy

# Plot the energy bands
plt.figure()
plt.hlines(energies, 0, 1, linewidth=1)

# Highlight central band if even number of sites
if n % 2 == 0:
    # Assume n+1 distinct energy bands (empirically)
    center_band = identify_center_band(energies, n+1)
    plt.hlines(center_band, 0, 1, linewidth=1, colors='orange',
        label='Central band')

# Draw Fermi level
plt.hlines(Efermi, 0, 1, colors='red', linestyles='--', linewidth=1,
    label=r'$E_F$')
plt.ylabel('E')
plt.legend(loc='best')
plt.title(f'Energy bands (n = {n}, U/t = {U/t:g})')

# Save plot
U_str = f'{U:g}'.replace('.', '_')
t_str = f'{t:g}'.replace('.', '_')
fig_file = f"fermi_hubbard_bands_n{n}_U{U_str}_t{t_str}.png"
print(f'Saving figure to file "{fig_file}"')
plt.savefig(fig_file)

# Zoom in on the conduction band if even number of sites
if n % 2 == 0:
    pad = .05   # Pad the axis range by 5%
    bandwidth = np.max(center_band) - np.min(center_band)
    band_middle = (np.min(center_band) + np.max(center_band)) / 2
    plt.ylim((band_middle - bandwidth*(0.5 + pad),
        band_middle + bandwidth*(0.5 + pad)))
    fig_file = f"fermi_hubbard_bands_zoomed_n{n}_U{U_str}_t{t_str}.png"
    print(f'Saving figure to file "{fig_file}"')
    plt.savefig(fig_file)

plt.show()
