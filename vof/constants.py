"""Shared physical constants and simulation unit conversions.

Numerical tolerances (eps) are intentionally kept local to each module, since they differ
per algorithm. convert_to_fits.py keeps its own unit-aware (unyt) constants.
"""

import numpy as np

pi = np.pi

# Simulation units (FIRE/GIZMO). Factors of h are accounted for in readsnap.
unit_M = 10**10 * 1.98855 * 10**33  # 10^10 solar masses, in grams
unit_L = 3.086 * 10**21  # 1 kpc, in cm
unit_V = 1.0 * 10.0**5  # 1 km/s, in cm/s. Converted by a factor of sqrt(a) in readsnap
unit_T = unit_L / unit_V
unit_rho = unit_M / unit_L**3

# Particle masses, in simulation mass units
proton_mass = 1.6726219 * 10 ** (-27) * (1000.0 / unit_M)  # kg -> g -> unit mass
amu = 1.6735575 * np.power(10.0, -24) / unit_M  # g -> unit mass
atomsPerMol = 6.02214 * np.power(10.0, 23.0)

# Physical constants (cgs / eV)
h = 4.135667696 * np.power(10.0, -15.0)  # Planck constant, eV * s
m_e = 9.1094 * np.power(10.0, -28.0)  # electron mass, g
e = 4.8032 * np.power(10.0, -10.0)  # electron charge, cm^(3/2) * g^(1/2) * s^(-1)
kb = 8.617333262145 * np.power(10.0, -5.0)  # Boltzmann constant, eV / K
c = 3.0 * np.power(10.0, 10.0)  # speed of light, cm/s
c_km_s = 3 * 10**5  # speed of light, km/s

# Angles
arcsec = (1.0 / 60.0 / 60.0) * pi / 180.0  # arcseconds -> radians
arcsec2rad = pi / (180.0 * 3600.0)  # same value as arcsec

# Lengths
kpc2cm = 3.086 * np.power(10.0, 21.0)
meters2Kpc = 3.24078 * np.power(10.0, -20.0)

# Flux units
Joules2eV = 6.241509 * np.power(10.0, 18.0)
Jy2SimUnits = (
    np.power(10.0, -26.0) * Joules2eV / meters2Kpc / meters2Kpc
)  # J * s^-1 * m^-2 * Hz^-1
SimUnits2Jy = 1.0 / Jy2SimUnits
