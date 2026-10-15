import h5py
import numpy as np

from vof.constants import (
    proton_mass,
    unit_L,
    unit_M,
    unit_T,
)
from vof.io.readsnap_vof import readsnap_initial, readsnap_trunc

####


eps = 1e-10

####Various IO functions for reading in the simulation snapshots, sightline information, and galaxy stats
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023


def read_stats(statsDir):
    # Read in previously saved centering information for a galaxy
    hf = h5py.File(statsDir, "r")
    posCenter = np.zeros(3)
    velCenter = np.zeros(3)
    Lhat = np.zeros(3)
    r0 = np.zeros(3)
    orientation_maxima = np.zeros(3)

    posCenter[:] = hf["pos_center"][:]
    velCenter[:] = hf["vel_center"][:]
    Lhat[:] = hf["Lhat"][:]
    r0[:] = hf["r0"][:]
    orientation_maxima[0] = np.array(hf["max_x"])
    orientation_maxima[1] = np.array(hf["max_y"])
    orientation_maxima[2] = np.array(hf["max_z"])

    hf.close()
    return posCenter, velCenter, Lhat, r0, orientation_maxima


def load_data(snapdir, Nsnapstring, ptype, rTrunc, posCenter, velCenter):
    # Load position and density to create a mask for particles in user defined region of interest
    particles = readsnap_initial(
        snapdir,
        Nsnapstring,
        ptype,
        snapshot_name="snapshot",
        extension=".hdf5",
        h0=1,
        cosmological=1,
    )  ##!! EDIT READSNAP INITIAL TO ONLY LOAD POS
    pos = particles["p"]  # positions

    truncMax = posCenter + rTrunc
    truncMin = posCenter - rTrunc

    truncMask = (
        (pos[:, 0] < truncMax[0])
        & (pos[:, 0] > truncMin[0])
        & (pos[:, 1] < truncMax[1])
        & (pos[:, 1] > truncMin[1])
        & (pos[:, 2] < truncMax[2])
        & (pos[:, 2] > truncMin[2])
    )

    pos = pos[truncMask]

    # Load the rest of the data only in the ROI to save memory
    particles = readsnap_trunc(
        snapdir,
        Nsnapstring,
        ptype,
        truncMask,
        snapshot_name="snapshot",
        extension=".hdf5",
        h0=1,
        cosmological=1,
    )  ##!! EDIT TO LOAD DENS AS WELL
    vel = particles["v"]
    mass = particles["m"]

    # Center the Data on galactic center
    pos -= posCenter
    vel -= velCenter

    if ptype == 0:
        dens = particles["rho"]
        metallicity = particles["z"]
        neutral_H_frac = particles["nh"]
        kernel_lengths = particles["h"]
        temp = calc_temps(particles["u"], particles["ne"], metallicity)

        return pos, vel, mass, dens, metallicity, neutral_H_frac, kernel_lengths, temp

    else:
        return pos, vel, mass


def calc_temps(GintE, ElectronAbundance, Gz):  # Calculate Temperatures
    gamma = 1.6666666666666666667
    kb = (
        1.38064852
        * 10 ** (-23)
        * (100.0 / unit_L) ** 2
        * (1000.0 / unit_M)
        * (unit_T) ** 2
    )  ## Boltzmann's constant, appropriate units
    # GintE = G['u'] #internal energy
    # ElectronAbundance = G['ne']
    # Gz = G['z']
    N = np.size(GintE)

    helium_mass_fraction = np.zeros(N)
    helium_mass_fraction[:] = Gz[:, 1]  # Gz columns: [total metallicity, helium, ...]

    y_helium = np.divide(
        helium_mass_fraction, np.multiply(4, np.subtract(1, helium_mass_fraction))
    )
    # y_helium = helium_mass_fraction / (4*(1-helium_mass_fraction))
    mu = np.divide(
        np.add(1, np.multiply(4, y_helium)),
        np.add(1, np.add(y_helium, ElectronAbundance)),
    )
    # mu = (1 + 4*y_helium) / (1+y_helium+ElectronAbundance)
    mean_molecular_weight = np.multiply(mu, proton_mass)
    # mean_molecular_weight = mu*proton_mass
    Gtemp = np.multiply(mean_molecular_weight, np.multiply(((gamma - 1) / kb), GintE))
    # Gtemp = mean_molecular_weight * (gamma-1) * GintE / kb

    return Gtemp


def load_data_for_image_gen(particles, posCenter, velCenter):
    # Center the gas particles (as loaded by binfire.load_gas) on the galaxy. Modifies particles["p"]/["v"] in place.
    pos = particles["p"]
    vel = particles["v"]
    pos -= posCenter
    vel -= velCenter
    return pos, particles["h"], vel


def load_masked_data_for_image_gen(particles, gKernel, species):
    # Mass, temperature, and emitting-species mass fraction for the gas particles (as loaded by binfire.load_gas)
    fHe = particles["z"][:, 1]
    z = particles["z"][:, 0]

    if "temp" in particles:
        temp = particles[
            "temp"
        ]  # Precomputed by load_gas when available, rather than recalculating here
    else:
        temp = calc_temps(particles["u"], particles["ne"], particles["z"])

    if species == "HI_21cm":
        speciesMassFrac = calc_molecular_fraction(
            particles["nh"], gKernel, particles["rho"], fHe, z, to_return="fH1"
        )

    return particles["m"], temp, speciesMassFrac


def calc_molecular_fraction(
    Gnh,
    KernelLengths,
    density,
    fHe,
    fMetals,
    to_return="fH1",
):
    # Inputs = Mass, number density, Kernel Length, density, metallicity
    Z = fMetals  # metal mass (everything not H, He)
    mu_H = (
        2.3 * np.power(10.0, -27.0) * (1000.0 / unit_M)
    )  #'average' mass of hydrogen nucleus from Krumholz&Gnedin 2011

    if to_return == "fHion":
        return (1.0 - fHe - fMetals) * (1.0 - Gnh)

    Z_MW = 0.02  # Assuming MW metallicity is ~solar on average (Check!!)
    Z_solar = 0.02  # From Gizmo documentation

    N_ngb = 32.0
    sobColDens = np.multiply(KernelLengths, density) / np.power(
        N_ngb, 1.0 / 3.0
    )  # Cheesy approximation of Column density

    tau = (
        np.multiply(sobColDens, Z)
        * 1
        / (mu_H * Z_MW)
        * np.power(10.0, -21.0)
        / (unit_L ** (2))
    )  # cm^2 to Unit_L^2
    tau[tau == 0] = eps  # avoid divide by 0

    chi = 3.1 * (1 + 3.1 * np.power(Z / Z_solar, 0.365)) / 4.1  # Approximation

    s = np.divide(np.log(1 + 0.6 * chi + 0.01 * np.power(chi, 2)), (0.6 * tau))
    s[s == -4.0] = -4.0 + eps  # Avoid divide by zero
    fH2 = np.divide(
        (1 - 0.5 * s), (1 + 0.25 * s)
    )  # Fraction of Molecular Hydrogen from Krumholz & Knedin

    lowerApproxLimit = 0.1
    denom = 7.2 * tau
    denom[denom == 0] = eps
    fH2[fH2 < lowerApproxLimit] = (
        1.0
        / 3.0
        * (
            2.0
            - 44.0
            * 2.5
            * 0.066
            / 7.2
            * np.divide(chi[fH2 < lowerApproxLimit], tau[fH2 < lowerApproxLimit])
        )
    )

    fH2[fH2 < 0] = 0  # Nonphysical negative molecular fractions set to 0

    if to_return == "fH1":
        fH1 = np.multiply((1.0 - fHe - fMetals), (1.0 - fH2 - (1.0 - Gnh)))
        fH1[fH1 < 0] = 0
        return fH1
    elif to_return == "fH2":
        return (1.0 - fHe - fMetals) * fH2
