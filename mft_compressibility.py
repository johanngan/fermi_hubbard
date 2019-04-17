#!/usr/bin/env python3
"""
Computes compressibility d(rho)/d(mu) in a 1D Fermi-Hubbard system using a mean
field approximation

Created March 19, 2019
by Johann Gan
"""

import numpy as np
import matplotlib.pyplot as plt
from energy_curves import calc_magnetization, calc_min_energy_afm

def calc_density_fm(nsites, U, t, mu, nup0=None, ndown0=None, max_iter=100):
    """
    Calculates density under a given chemical potential, assuming
    ferromagnetism.

    Inputs:
        nsites = number of sites
        U, t, mu = Hubbard parameters
        nup0, ndown0 (default None) = initial guess for total particle numbers
            If None, generated randomly
        max_iter (default 100) = maximum number of iterations before terminating
    Outputs:
        density, energy, nup, ndown = final density, energy, number of spin-up
            and spin-down particles
        icount = number of mean-field iterations to converge
        phase = 1 if final magnetization is "big enough", else 0
    """
    nup_old = None
    ndown_old = None
    # Pick random initial condition if none is specified
    if nup0 is None:
        nup = np.random.randint(nsites+1)
    else:
        nup = nup0
    if ndown0 is None:
        ndown = np.random.randint(nsites)
    else:
        ndown = ndown0

    # particle-hole symmetric form: effective mu becomes mu + U/2
    mu += U/2

    # momentum states
    allowed_momenta = np.linspace(-np.pi + 2*np.pi/nsites, np.pi, nsites)
    # kinetic energy in ascending order
    allowed_kinetic_energies = np.sort(-2*t*np.cos(allowed_momenta))

    # Iteratively update particle numbers until self-consistent with mean field
    icount = 0
    converged = True
    # For keeping track of whether the iteration bounces between two values forever
    nup_oldold = None
    ndown_oldold = None
    while nup_old != nup or ndown_old != ndown:
        icount += 1
        if icount > max_iter:
            converged = False
            break
        # Costs for cumulatively adding the next spin-up or spin-down particles
        up_energy_costs = allowed_kinetic_energies + U*ndown/nsites
        down_energy_costs = allowed_kinetic_energies + U*nup/nsites

        # Add particles until the energy cost outweights the energy decrease
        # due to the chemical potential
        nup_oldold = nup_old
        ndown_oldold = ndown_old
        nup_old = nup
        ndown_old = ndown
        nup = len(np.where(up_energy_costs < mu)[0])
        ndown = len(np.where(down_energy_costs < mu)[0])
        # Average with the old to stabilize oscillatory behavior
        weight = 2/3
        nup = round(weight*nup + (1-weight)*nup_old)
        ndown = round(weight*ndown + (1-weight)*ndown_old)
        # Got stuck in an infinite loop -- average the two and call it a day
        if nup_oldold == nup and ndown_oldold == ndown:
            nup = (nup + nup_old) // 2  # Force to be integer
            ndown = (ndown + ndown_old) // 2
            # converged = False
            break
    # Calculate energy density (per site) of the settled configuration
    density = (nup+ndown) / nsites
    energy = (np.sum(allowed_kinetic_energies[:nup]) \
        + np.sum(allowed_kinetic_energies[:ndown])) / nsites \
        + U * nup*ndown / nsites**2 \
        - mu * density
    # Check if FM or PM phase
    resolution = 2/(nup + ndown) if nup + ndown > 2 else 0
    if abs(calc_magnetization(nup/nsites, ndown/nsites)) <= resolution:
        phase = 0   # PM
    else:
        phase = 1   # FM
    # Return density, and also raw particle numbers + iteration count
    return density, energy, nup, ndown, icount, converged, phase

def calc_density_afm(nsites, U, t, mu, tol=1e-2, n0=None, m0=None, max_iter=100):
    """
    Calculates density under a fixed chemical potential, assuming
    antiferromagnetism.

    Inputs:
        nsites = number of sites
        U, t, mu = Hubbard parameters
        tol = tolerance for convergence of m
        n0, m0 (default None) = initial guess for total particle number and
            staggered magnetization
        max_iter (default 100) = maximum number of mean-field iterations before
            terminating
    Outputs:
        density, energy, n, m = final density, energy, particle number, and
            staggered magnetization
        icount = number of mean-field iterations to converge
    """
    # particle-hole symmetric form: effective mu becomes mu + U/2
    mu += U/2

    n_old = None
    m_old = None
    # Pick random initial condition if none is specified
    if n0 is None:
        n = 2*np.random.randint(nsites+1)
    else:
        n = n0
    if m0 is None:
        m = np.random.uniform(0, 1)
    else:
        m = m0

    # momentum states
    allowed_momenta = np.linspace(-np.pi/2 + 2*np.pi/nsites, np.pi/2, nsites/2)

    # Iteratively update particle numbers until self-consistent with mean field
    icount = 0
    converged = True
    # For keeping track of whether the iteration bounces between two values forever
    n_oldold = None
    while n != n_old or abs(m - m_old) > tol:
        icount += 1
        if icount > max_iter:
            converged = False
            break

        # Evenly distributed between spin-up and spin-down
        density = n / (2*nsites)
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

        n_oldold = n_old
        n_old = n
        m_old = m
        # Update n
        n = 2*len(np.where(allowed_energies < mu)[0])
        # Average with the old to stabilize oscillatory behavior
        weight = 2/3
        n = round(weight*n + (1-weight)*n_old)
        n = (n // 2) * 2    # Force to be even
        # Update m
        m, energy = calc_min_energy_afm(nsites, n, U, t)

        # Got stuck in an infinite loop -- average the two and call it a day
        if n_oldold == n:
            n = (n + n_old) / 2
            n = (n // 2) * 2    # Force to be even
            m, energy = calc_min_energy_afm(nsites, n, U, t)
            # converged = False
            break
    # Calculate energy density (per site) of the settled configuration
    density = n / nsites
    energy -= mu * density
    # Check if AFM or PM phase
    resolution = 2/n if n > 2 else 0
    phase = 0 if abs(m) <= resolution else -1

    # Return density, and also raw particle number, magnetization + iteration count
    return density, energy, n, m, icount, converged, phase

def calc_density_mc(calc_density, nsites, U, t, mu, trials=20):
    """
    Compute the density for a given chemical potential using a Monte-Carlo
    method to sample different initial guesses, then picking the lowest-energy
    configuration found.

    Inputs:
        calc_density = function that makes a single guess for density under
            a given chemical potential
        nsites = number of sites
        U, t, mu = Hubbard parameters
        trials (default 20) = number of initial guesses to try before picking
            the best
    Outputs:
        best_density, best_energy, best_phase = density and corresponding
            energy and phase of the lowest-energy configuration found
        succeeded = flag for whether or not the mean-field calculation
            converged
    """
    best_density = None
    best_energy = np.inf
    best_phase = None
    succeeded = False
    for i in range(trials):
        density, energy, _, _, _, converged, phase = calc_density(nsites, U, t, mu)
        succeeded = succeeded or converged
        if energy < best_energy:
            best_density = density
            best_energy = energy
            best_phase = phase
    return best_density, best_energy, best_phase, succeeded

def derivative(yvals, xvals):
    """
    Approximates the derivative dy/dx at each point in data arrays

    Inputs:
        yvals, xvals = y and x data arrays. Must be at least two values long.
    Outputs:
        dydx = estimated derivatives at each point
    """
    dydx = np.zeros(yvals.shape)    # Initialize
    # Forward difference for first entry
    dydx[0] = (yvals[1] - yvals[0]) / (xvals[1] - xvals[0])
    # Backward difference for last entry
    dydx[-1] = (yvals[-1] - yvals[-2]) / (xvals[-1] - xvals[-2])
    # Central difference for middle entries
    dydx[1:-1] = (yvals[2:] - yvals[:-2]) / (xvals[2:] - xvals[:-2])
    return dydx

def plot_mc_results(ax, x, y, successes, phases):
    """
    Plots y vs. x on given axes, marks x values that failed to converge,
    and color points based on the phase.

    Inputs:
        ax = axes to plot on
        x, y = data to plot
        successes = boolean array for successful convergence or not
        phases = list of phases 1, 0, -1 (FM, PM, AFM) of each data point
    """
    # Plot the rho vs. mu curve
    ax.plot(x, y)
    # Mark with a red line if the value isn't a fully converged estimate
    ax.vlines(x[np.where(np.logical_not(successes))], np.min(y), np.max(y),
        colors='r', linestyles='--')
    # Overlay the phase with color-coded dots
    ax.plot(x[np.where(phases == 1)], y[np.where(phases == 1)], 'ob',
        label='FM')
    ax.plot(x[np.where(phases == 0)], y[np.where(phases == 0)], 'og',
        label='PM')
    ax.plot(x[np.where(phases == -1)], y[np.where(phases == -1)], 'or',
        label='AFM')

# System parameters
nsites = 256
U = 4
t = 1
mus = np.linspace(-6, 6, 21)

# Functions for FM/AFM simulation
density_fn_fm = lambda mu: calc_density_mc(calc_density_fm, nsites, U, t, mu)
density_fn_afm = lambda mu: calc_density_mc(calc_density_afm, nsites, U, t, mu)

# Initialize arrays
# Ferromagnetic assumption
densities_fm = np.zeros(mus.shape)
energies_fm = np.zeros(mus.shape)
phases_fm = np.zeros(mus.shape)
successes_fm = np.zeros(mus.shape)
# Antiferromagnetic assumption
densities_afm = np.zeros(mus.shape)
energies_afm = np.zeros(mus.shape)
phases_afm = np.zeros(mus.shape)
successes_afm = np.zeros(mus.shape)
# Optimal case between FM/AFM
densities_opt = np.zeros(mus.shape)
energies_opt = np.zeros(mus.shape)
phases_opt = np.zeros(mus.shape) # 1 = FM, -1 = AFM, 0 = PM
successes_opt = np.zeros(mus.shape)

for i, mu in enumerate(mus):
    print('\033[K', end='') # Clear line
    print(f'Simulating mu = {mu:g} ({i+1}/{len(mus)})', end='\r')
    # Calculate and record
    densities_fm[i], energies_fm[i], phases_fm[i], successes_fm[i] = density_fn_fm(mu)
    densities_afm[i], energies_afm[i], phases_afm[i], successes_afm[i] = density_fn_afm(mu)

    # Decide on which assumed phase is optimal
    if energies_fm[i] < energies_afm[i]: # FM is better
        densities_opt[i] = densities_fm[i]
        energies_opt[i] = energies_fm[i]
        phases_opt[i] = phases_fm[i]
        successes_opt[i] = successes_fm[i]
    else:   # AFM is better or the same
        densities_opt[i] = densities_afm[i]
        energies_opt[i] = energies_afm[i]
        phases_opt[i] = phases_afm[i]
        successes_opt[i] = successes_afm[i]
print('\n')

# Compute compressibility d(rho)/d(mu)
compressibility = derivative(densities_opt, mus)

figsize = (10, 6)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
fig.suptitle('Relation between chemical potential and occupation\n' +
    f'(U = {U:g}, t = {t:g})')
# Plot the rho vs. mu curve
plot_mc_results(ax1, mus, densities_opt, successes_opt, phases_opt)
ax1.set_xlabel(r'$\mu$')
ax1.set_ylabel(r'$\rho$')
ax1.set_title(r'$\rho$ vs. $\mu$')
ax1.legend(loc='best')
# Plot the compressibility curve
plot_mc_results(ax2, mus, compressibility, successes_opt, phases_opt)
ax2.set_xlabel(r'$\mu$')
ax2.set_ylabel(r'$d\rho/d\mu$')
ax2.set_title('Compressibility')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save plot
U_str = f'{U:g}'.replace('.', '_')
t_str = f'{t:g}'.replace('.', '_')
fig_file = f"fermi_hubbard_compressibility_U{U_str}_t{t_str}.png"
print(f'Saving figure to file "{fig_file}"')
plt.savefig(fig_file)

plt.show()
