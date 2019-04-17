# Simulation of the 1D Fermi-Hubbard model

This project simulates three different properties of spin-1/2 fermions on a 1D lattice. The file `fermi_hubbard_report.pdf` contains a full writeup.

## Small-system band structure calculation

`ed.py` performs exact diagonalization to calculate the energy bands for very small systems (N <= 7) with periodic boundary conditions. Plots the band structure diagram with the Fermi level and highlights the conduction band if any.

## Magnetic phase diagram

`mft_phase.py` uses mean field theory to calculate the magnetic phase diagram over different densities and interaction strengths.

## Compressibility

`mft_compressibility.py` uses self-consistent mean field theory and Monte-Carlo sampling to compute the dependence of density on external chemical potential, and plots it with the phase associated with each density. Also computes the compressibility (d(rho)/d(mu)), an easier-to-calculate "stand-in" for conductivity.
