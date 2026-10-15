import numpy as np
from scipy import stats

####Function to calculate the rotational velocity curve of a galaxy. Used for placing a mock observer within a galaxy
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023


def calc_rotation_curve(pos, vel, mass, nr, max_r):
    rMag = np.linalg.norm(pos, axis=1)
    rHat = np.copy(pos)
    rHat[:, 0] = np.divide(pos[:, 0], rMag)
    rHat[:, 1] = np.divide(pos[:, 1], rMag)
    rHat[:, 2] = np.divide(pos[:, 2], rMag)
    vMag = np.linalg.norm(
        vel - np.multiply(vel, rHat), axis=1
    )  # Don't count radial velocity

    # Calculate mass weighted average
    binned_mom, _binedge1d, _binnum1d = stats.binned_statistic_dd(
        rMag, np.multiply(vMag, mass), "sum", nr, range=[[0, max_r]]
    )
    binned_mass, _binedge1d, _binnum1d = stats.binned_statistic_dd(
        rMag, mass, "sum", nr, range=[[0, max_r]]
    )

    return np.divide(binned_mom, binned_mass)
