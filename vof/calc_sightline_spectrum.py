"""Calculate the spectrum along a single sightline of finite beam size.

Finds all particles that intersect the sightline and sorts them by distance, such that the
full spectrum can be built up by accounting for emission then absorption starting with the
furthest particle. Calls generate_spectra to determine the emission and full spectrum from
all particles in the sightline.

Written by Cameron Trapp (ctrapped@gmail.com).
"""

import numpy as np

from vof.generate_spectra import generate_spectra

eps = 1e-10


def calc_sightline_spectrum(
    sightline,
    testPositions,
    testVel,
    testRadii,
    testMass,
    testTemp,
    testSpeciesMassFrac,
    beamSize,
    speciesToRun,
    Nspec,
    bandwidth,
    rObserver,
    calcThermalLevels,
):  # assume already centered on observer
    zHat = sightline / np.linalg.norm(
        sightline
    )  # orient cylindrical coordinate system along the sightline
    N, _dim = np.shape(testPositions)
    r_z = np.zeros((N, 3))  # vector for the z component in the old coord system

    zmag = np.zeros(N)
    zmag = np.dot(testPositions, zHat)
    r_z[:, 0] = zmag * zHat[0]
    r_z[:, 1] = zmag * zHat[1]
    r_z[:, 2] = zmag * zHat[2]

    r_s = np.zeros((N, 3))
    r_s[:, :] = np.subtract(
        testPositions[:, :], r_z[:, :]
    )  # vector for the s component in the old coord system
    del testPositions
    del r_z
    smag = np.zeros(N)
    smag[:] = np.linalg.norm(r_s, axis=1)
    del r_s

    beamRadiusPhysical = np.zeros(N)
    beamRadiusPhysical[:] = (
        zmag[:] * beamSize / 2
    )  # the physical size of the beam for each particle

    mask = np.where(((smag - testRadii - beamRadiusPhysical) < 0) & (zmag > 0))

    testMass = testMass[mask]
    testSpeciesMassFrac = testSpeciesMassFrac[mask]
    testVel = testVel[mask]
    testRadii = testRadii[mask]
    testTemp = testTemp[mask]
    zmag = zmag[mask]
    smag = smag[mask]
    del mask

    dopplerVelocity = np.dot(testVel, zHat)
    dopplerVelocity = np.add(
        dopplerVelocity, (zmag - rObserver) * 0.07
    )  # kpc * km/s /kpc, Hubble flow but still centered on galaxy

    indices = np.argsort(zmag)  # start with the closest particles

    # Calculate the spectrum and emission from these particles
    spectrum, emission, _optical_depth, _nu = generate_spectra(
        testMass[indices],
        testSpeciesMassFrac[indices],
        dopplerVelocity[indices],
        testRadii[indices],
        testTemp[indices],
        zmag[indices],
        smag[indices],
        species=speciesToRun,
        beamSize=beamSize,
        Nspec=Nspec,
        bandwidth=bandwidth,
        calcThermalLevels=calcThermalLevels,
    )

    return spectrum, emission
