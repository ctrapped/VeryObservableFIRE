"""Center on the observer and orient coordinate systems along given vectors.

Written by Cameron Trapp (ctrapped@gmail.com).
"""

import math

import numpy as np
from scipy import interpolate

eps = 0.00000000000000000000000000000000000000000000001


def center_on_observer(
    observer_position, observer_velocity=None, rotationCurve=None, max_r=None
):
    """Return the observer's Cartesian position and velocity.

    If observer_velocity is None, the observer moves azimuthally at the rotation-curve
    speed.
    """
    # Observer position and velocity in (pseudo)cylindrical coordinates?
    r, phi, z = observer_position
    pos_observer = [r * np.cos(phi), r * np.sin(phi), z]

    # Define Observer Velocity from RC if not provided
    if observer_velocity is None:
        # Assume only moving azimuthally
        rInterp = np.linspace(0, max_r, np.size(rotationCurve), endpoint=True)
        f = interpolate.interp1d(rInterp, rotationCurve)
        observer_velocity = [0, f(r), 0]

    vel_observer = np.zeros(3)
    vr, vphi, vz = observer_velocity
    vel_observer[0] = vr * math.cos(phi) - vphi * math.sin(phi)
    vel_observer[1] = vr * math.sin(phi) + vphi * math.cos(phi)
    vel_observer[2] = vz

    return pos_observer, vel_observer


def orient_galaxy(pos, vel, Lhat, r0, returnRotationalVelocity=False):
    """Rotate positions (and velocities) so Lhat is the z axis and r0 the x axis.

    Returns new arrays; the inputs are not modified.
    """
    xhat = r0
    zhat = Lhat
    yhat = np.cross(zhat, xhat)

    new_pos = np.empty_like(pos)
    new_pos[:, 0] = (
        np.multiply(pos[:, 0], xhat[0])
        + np.multiply(pos[:, 1], xhat[1])
        + np.multiply(pos[:, 2], xhat[2])
    )
    new_pos[:, 1] = (
        np.multiply(pos[:, 0], yhat[0])
        + np.multiply(pos[:, 1], yhat[1])
        + np.multiply(pos[:, 2], yhat[2])
    )
    new_pos[:, 2] = (
        np.multiply(pos[:, 0], zhat[0])
        + np.multiply(pos[:, 1], zhat[1])
        + np.multiply(pos[:, 2], zhat[2])
    )
    if vel is not None:
        new_vel = np.empty_like(vel)
        new_vel[:, 0] = (
            np.multiply(vel[:, 0], xhat[0])
            + np.multiply(vel[:, 1], xhat[1])
            + np.multiply(vel[:, 2], xhat[2])
        )
        new_vel[:, 1] = (
            np.multiply(vel[:, 0], yhat[0])
            + np.multiply(vel[:, 1], yhat[1])
            + np.multiply(vel[:, 2], yhat[2])
        )
        new_vel[:, 2] = (
            np.multiply(vel[:, 0], zhat[0])
            + np.multiply(vel[:, 1], zhat[1])
            + np.multiply(vel[:, 2], zhat[2])
        )
        if returnRotationalVelocity:
            j = np.cross(pos, vel)
            jz = (
                np.multiply(j[:, 0], zhat[0])
                + np.multiply(j[:, 1], zhat[1])
                + np.multiply(j[:, 2], zhat[2])
            )
            rmag = np.linalg.norm(pos, axis=1)
            rotVel = np.divide(jz, rmag)
            return new_pos, new_vel, rotVel

        return new_pos, new_vel
    else:
        return new_pos


def get_radial_velocity(pos, vel, rObs):
    """Return each particle's radial velocity plus Hubble flow relative to rObs."""
    rMag = np.linalg.norm(pos, axis=1)
    rHat = np.copy(pos)
    rHat[:, 0] = np.divide(pos[:, 0], rMag)
    rHat[:, 1] = np.divide(pos[:, 1], rMag)
    rHat[:, 2] = np.divide(pos[:, 2], rMag)

    rVel = (
        np.multiply(vel[:, 0], rHat[:, 0])
        + np.multiply(vel[:, 1], rHat[:, 1])
        + np.multiply(vel[:, 2], rHat[:, 2])
    )

    rVel = np.add(
        rVel, (rMag - rObs) * 0.07
    )  # kpc * km/s /kpc, Hubble flow but still centered on galaxy

    return rVel
