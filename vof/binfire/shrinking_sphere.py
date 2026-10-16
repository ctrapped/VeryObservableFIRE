"""Shrinking-sphere fit for the center of a galaxy's stellar distribution.

Written by Cameron Trapp (ctrapped@gmail.com).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def center_of_mass(Smas, r_s):
    """Return the mass-weighted center of the given positions."""
    pos_com = np.zeros(3)
    pos_com[:] = np.divide(np.dot(np.transpose(r_s), Smas), np.sum(Smas))
    logger.debug("New center of mass: %s", pos_com)
    return pos_com


def shrink_sphere(
    Smas, Spos, wiggle_count, r=None, rmin=10, shrink_factor=0.7, old_com=None
):
    """Find the center of mass with a shrinking-sphere fit.

    Shrinks the sphere by shrink_factor each iteration down to rmin, then re-centers
    until the center moves less than delta_cutoff. Recommended with stellar mass, at
    least for m12 galaxies.
    """
    delta_cutoff = 0.000001

    # Shrinking phase: recompute the c.o.m. and filter to particles within a shrinking radius, until the radius reaches rmin
    while True:
        NS = np.size(Smas)
        logger.debug("%d stars in sphere", NS)

        logger.debug("Calculating new center of mass")
        pos_com = center_of_mass(Smas, Spos)

        r2_S = np.add(
            np.power(Spos[:, 0] - pos_com[0], 2),
            np.add(
                np.power(Spos[:, 1] - pos_com[1], 2),
                np.power(Spos[:, 2] - pos_com[2], 2),
            ),
        )

        r_new = r * shrink_factor
        logger.debug("New radius: %s", r_new)

        if r_new > rmin:  # iterate until few enough particles
            # Set a min radius, #Iterate at the minimum radius until the com is stable
            logger.debug("Iterating again")
            Sidx = np.where(r2_S < np.power(r_new, 2))
            del r2_S

            Smas = Smas[Sidx]
            Spos = Spos[Sidx]
            r = r_new
            old_com = pos_com
        else:
            Sidx = np.where(r2_S < rmin**2)
            del r2_S
            break

    # Wiggling phase: keep re-centering within the fixed rmin sphere until the c.o.m. stabilizes
    Smas = Smas[Sidx]
    Spos = Spos[Sidx]
    final_com = center_of_mass(Smas, Spos)
    delta_com = np.subtract(final_com, old_com)
    delta = np.linalg.norm(delta_com)

    while ((delta > delta_cutoff) and (wiggle_count < 1000)) or (
        (wiggle_count < 10) and (delta != 0.0)
    ):
        logger.debug("Wiggling the sphere around")
        logger.debug("Delta C.O.M. = %s", delta)
        wiggle_count += 1

        NS = np.size(Smas)
        logger.debug("%d stars in sphere", NS)
        logger.debug("Calculating new center of mass")
        pos_com = center_of_mass(Smas, Spos)
        r2_S = np.add(
            np.power(Spos[:, 0] - pos_com[0], 2),
            np.add(
                np.power(Spos[:, 1] - pos_com[1], 2),
                np.power(Spos[:, 2] - pos_com[2], 2),
            ),
        )
        Sidx = np.where(r2_S < rmin**2)
        del r2_S

        Smas = Smas[Sidx]
        Spos = Spos[Sidx]
        old_com = np.copy(final_com)
        final_com = center_of_mass(Smas, Spos)
        delta_com = np.subtract(final_com, old_com)
        delta = np.linalg.norm(delta_com)

    logger.debug("Delta C.O.M. = %s", delta)
    logger.debug("Shrinking sphere converged after %d wiggles", wiggle_count)

    return final_com
