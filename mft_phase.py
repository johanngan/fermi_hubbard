#!/usr/bin/env python3
"""
Creates a magnetic phase diagram for the 1D Fermi-Hubbard model using a mean
field approximation

Created March 19, 2019
by Johann Gan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from energy_curves import (
    calc_energy_curve_fm, calc_energy_curve_afm,
    calc_min_energy_fm, calc_min_energy_afm
)

def determine_phase(nsites, nparticles, U, t):
    """
    Determines whether a system will exhibit ferromagnetic, antiferromagnetic,
    or paramagnetic order (which has the lowest energy)

    Inputs:
        nsites = number of sites
        nparticles = total number of particles in the system
        U, t = Hubbard parameters
    Outputs:
        1 if ferromagnetic, -1 if antiferromagnetic, 0 if paramagnetic
    """
    
    # Discreteness of the cosine term in energy expressions leads to weird
    # artifact where the configuration with nup = ndown + 2 (m = 2/nparticles)
    # has slightly lower energy than nup = ndown (m = 0). This is basically 0
    # for a large number of particles.
    resolution = 2/nparticles if nparticles > 2 else 0
    magnetization, energy_fm = calc_min_energy_fm(nsites, nparticles, U, t)
    staggered_magnetization, energy_afm = calc_min_energy_afm(nsites,
        nparticles, U, t)
    if energy_fm <= energy_afm: # FM is better than AFM
        return 0 if abs(magnetization) <= resolution else 1, \
            abs(magnetization)
    else:   # AFM is better than FM
        return 0 if abs(staggered_magnetization) <= resolution else -1, \
            abs(staggered_magnetization)

def calc_phase_diagram(U_over_t_max=10, U_over_t_npoints=257, nparticle_res=2,
    nsites=512):
    """
    Calculates magnetic phase diagram and order parameters for the
    Fermi-Hubbard model over a range of densities and U/t values.

    Inputs:
        U_over_t_max (default 10) = maximum U/t value (minimum is 0)
        U_over_t_npoints (default 257) = number of U/t points to consider on
            the interval [0, U_over_t_max]
        nparticle_res (default 2) = step size when varying number of total
            particles. Should be even since AFM assumes equal numbers of
            spin-up and spin-down
        nsites (default 512) = number of sites
    Outputs:
        densities, U_over_ts = range of density and U/t values
        phase_grid, order_params = phase (0, +-1, see determine_phase function)
            and order parameter (magnetization/staggered magnetization) at each
            phase space point in a meshed grid of densities and U_over_ts
    """
    # All possible even particles numbers
    nparticles = np.arange(0, 2*nsites+1, nparticle_res)
    densities = nparticles / nsites

    # Range of U/t values
    U_over_ts = np.linspace(0, U_over_t_max, U_over_t_npoints)

    # Compute phase/order parameter at each phase space point
    nparticles_grid, U_over_t_grid = np.meshgrid(nparticles, U_over_ts)
    phase_func = np.vectorize(
        lambda nparticles, U_over_t: determine_phase(
            nsites, nparticles, U_over_t, 1))
    phase_grid, order_params = phase_func(nparticles_grid, U_over_t_grid)
    return densities, U_over_ts, phase_grid, order_params

def make_colormap(n, s=1, v=1):
    """
    Makes a colormap with n colors with the same saturation
    and value, but hue that varies from 0 to (n-1)/n (color wheel, basically).
    Inputs:
        n = number of colors in the map
        s, v (default 1) = saturation, value
    Outputs:
        cmap = colormap as specified
    """
    # Form the colors
    cols = np.ones((n, 3))
    cols[:, 0] = np.arange(n) / n
    return colors.ListedColormap(colors.hsv_to_rgb(cols))

# Calculate phase diagram - displays AFM near rho=1, and at rho=1 down to U = 0
# as expected by the duality with the Heisenberg model, J = 4t^2/U
U_over_t_max = 10
U_over_t_npoints = 257
nparticle_res = 2
nsites = 512
densities, U_over_ts, phase_grid, order_params = calc_phase_diagram(
    U_over_t_max, U_over_t_npoints, nparticle_res, nsites)

# Form discrete colormap for phases
cmap = make_colormap(len(np.unique(phase_grid)))
# Bins include bottom and exclude top: [a, b)
bounds = np.arange(np.min(phase_grid), np.max(phase_grid)+2)
# Discrete colormap intervals
norm = colors.BoundaryNorm(bounds, cmap.N)
# Define x and y ranges
extent = tuple((np.min(densities), np.max(densities))) \
    + tuple((np.min(U_over_ts), np.max(U_over_ts)))

# Set up plot
figsize = (12, 6)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

# Plot rho-U/t phase diagram
im1 = ax1.imshow(phase_grid, extent=extent, interpolation='none',
    origin='lower', aspect='auto', cmap=cmap, norm=norm)
ax1.set_title('Phases')
ax1.set_xlabel(r'$\rho$')
ax1.set_ylabel(r'U/t')
# Label each color by phase
cbar = plt.colorbar(im1, ax=ax1, cmap=cmap, boundaries=bounds, ticks=bounds)
labels = ['AFM', 'PM', 'FM']
cbar.ax.get_yaxis().set_ticks([])
for i, lab in enumerate(labels):
    cbar.ax.text(.5, i/len(labels) + 1/(2*len(labels)), lab,
        ha='center', va='center')
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_yticklabels(['AFM','PM','FM'])
cbar.set_label('Phase', rotation=270)

# Plot magnitude of the order parameters in the phase diagram
im2 = ax2.imshow(order_params, extent=extent, interpolation='none',
    origin='lower', aspect='auto')
ax2.set_title('Order parameter magnitude')
ax2.set_xlabel(r'$\rho$')
ax2.set_ylabel(r'U/t')
plt.colorbar(im2, ax=ax2)

# Finalize plot
plt.suptitle('Fermi-Hubbard model magnetic phase diagram')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save plot
fig_file = f"fermi_hubbard_phase_diagram.png"
print(f'Saving figure to file "{fig_file}"')
plt.savefig(fig_file)

plt.show()
