"""Calculate the rotation curve of a galaxy.

Used to give a mock observer placed inside the galaxy its orbital velocity.

Written by Cameron Trapp (ctrapped@gmail.com).
"""

import numpy as np
from scipy import stats


def calc_rotation_curve(pos, vel, mass, nr, max_r):
    """Return the mass-weighted mean non-radial speed in nr radial bins out to max_r."""
    rMag = np.linalg.norm(pos, axis=1)
    rHat = np.copy(pos)
    rHat[:, 0] = np.divide(pos[:, 0], rMag)
    rHat[:, 1] = np.divide(pos[:, 1], rMag)
    rHat[:, 2] = np.divide(pos[:, 2], rMag)
    v_radial = np.sum(vel * rHat, axis=1)  # component of each velocity along r-hat
    vMag = np.linalg.norm(
        vel - v_radial[:, np.newaxis] * rHat, axis=1
    )  # Don't count radial velocity

    # Calculate mass weighted average
    binned_mom, _binedge1d, _binnum1d = stats.binned_statistic_dd(
        rMag, np.multiply(vMag, mass), "sum", nr, range=[[0, max_r]]
    )
    binned_mass, _binedge1d, _binnum1d = stats.binned_statistic_dd(
        rMag, mass, "sum", nr, range=[[0, max_r]]
    )

    return np.divide(binned_mom, binned_mass)
