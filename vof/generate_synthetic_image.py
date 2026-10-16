"""Build one mock observation and write it to disk.

Forms the spectral datacube for one observer orientation (via project_image), then writes
the noisy and ideal cubes, plus the projected gas-property annotations when
projectGasProperties is set, to an HDF5 file with the observation setup as attributes.
Optionally also saves zeroth/first-moment and annotation PNG previews.

Written by Cameron Trapp (ctrapped@gmail.com).
"""

import logging
import os.path
import time

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from vof.constants import (
    arcsec,
    pi,
)
from vof.project_image import project_image

logger = logging.getLogger(__name__)

image = 0


def generate_synthetic_image(
    config,
    fileDir,
    statsDir,
    Nsnap,
    output,
    Lhat_rot,
    r0_rot,
    particleData=None,
):

    # Config values this function's own body needs. Everything else project_image needs
    # is read directly out of config there instead of being passed through here.
    observerDistance = config["observerDistance"]
    maxRadius = config["maxRadius"]
    beamSize = config["beamSize"]
    targetBeamSize = config["targetBeamSize"]
    Nsightlines1d = config["Nsightlines1d"]
    phiObs = config["phiObs"]
    Nspec = config["Nspec"]
    bandwidth_km_s = config["bandwidth_km_s"]
    savePNG = config["savePng"]
    projectGasProperties = config["projectGasProperties"]

    t1 = time.time()
    Nsnapstring = str(Nsnap)
    snapDir = fileDir + Nsnapstring  # Directory containing the actual snapshots
    statsDir = (
        statsDir + Nsnapstring.zfill(4) + ".hdf5"
    )  # Centering/orientation information.

    Nsightlines = Nsightlines1d * Nsightlines1d

    logger.debug("Generating sightline files")
    observer_position = np.array(
        [-observerDistance, phiObs, 0]
    )  # in spherical coordinates
    maxPhi = (
        2 * maxRadius / observerDistance
    )  # Convert physical size of observation to radians on the sky

    maxTheta = min(maxPhi, pi)

    maxima = [maxRadius, maxPhi, maxTheta]
    logger.debug("Pixel size: %.4g arcsec", beamSize / arcsec)

    # Predefine which particles belong to which sightline files to speed up parallelization. Can be re-used for observations from the same distance/inclination
    if projectGasProperties:
        (
            ideal_image,
            _smooth_image,
            noisy_image,
            mass_map,
            rMom_map,
            sMom_map,
            rotMom_map,
        ) = project_image(
            config,
            snapDir,
            Nsnapstring,
            statsDir,
            observer_position,
            maxima,
            Nsightlines,
            Lhat_rot,
            r0_rot,
            particleData=particleData,
        )
    else:
        ideal_image, _smooth_image, noisy_image = project_image(
            config,
            snapDir,
            Nsnapstring,
            statsDir,
            observer_position,
            maxima,
            Nsightlines,
            Lhat_rot,
            r0_rot,
            particleData=particleData,
        )

    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    hf = h5py.File(output + "_fullSpectra.hdf5", "w")
    hf.create_dataset("spectra", data=noisy_image)
    hf.create_dataset("ideal_image", data=ideal_image)
    # hf.create_dataset('smooth_image',data=smooth_image)
    hf.attrs["fov_kpc"] = 2 * maxRadius
    hf.attrs["observer_distance_kpc"] = observerDistance
    hf.attrs["beam_arcsec"] = targetBeamSize / arcsec
    if bandwidth_km_s is not None:
        hf.attrs["dnu_kmps"] = bandwidth_km_s / Nspec
    if projectGasProperties:
        mass_map[mass_map == 0] = 1e-20
        hf.create_dataset("mass_annotation", data=mass_map)
        hf.create_dataset(
            "radial_velocity_annotation", data=np.divide(rMom_map, mass_map)
        )
        hf.create_dataset(
            "cylindrial_radial_velocity_annotation", data=np.divide(sMom_map, mass_map)
        )
        hf.create_dataset(
            "rotational_velocity_annotation", data=np.divide(rotMom_map, mass_map)
        )
        hf.attrs["annotation_mass_units"] = "Msun"
        hf.attrs["annotation_velocity_units"] = "km/s"
    hf.close()

    if (
        savePNG
    ):  # Option to create a column density map to visualize results immediately
        plt.figure()
        m0 = np.sum(noisy_image, 2)
        vmax = np.max(m0)
        vmin = vmax * 1e-8
        plt.imshow(np.abs(m0), norm=LogNorm(vmin=vmin, vmax=vmax), cmap="inferno")
        plt.colorbar(label="Column Density [cm$^{-2}$]")
        plt.savefig(output + "_ZerothMomentMap.png")
        plt.close()

        plt.figure()
        spec = np.linspace(-bandwidth_km_s / 2, bandwidth_km_s / 2, Nspec)
        m1 = np.divide(
            np.sum(np.multiply(noisy_image, spec[None, None, :]), axis=2), m0
        )
        plt.imshow(
            m1, vmin=-bandwidth_km_s / 2, vmax=bandwidth_km_s / 2, cmap="seismic"
        )
        plt.colorbar(label="First Moment [km/s]")
        plt.savefig(output + "_FirstMomentMap.png")
        plt.close()

        if projectGasProperties:
            plt.figure()

            plt.imshow(mass_map, norm=LogNorm(), cmap="inferno")
            plt.colorbar(label="Projected Mass (Msun)")
            plt.savefig(output + "_ProjectedMass.png")
            plt.close()

            plt.figure()
            vmax = 150
            vmin = -vmax
            plt.imshow(
                np.divide(rMom_map, mass_map), vmin=vmin, vmax=vmax, cmap="seismic"
            )
            plt.colorbar(label="Radial Velocity (km/s)")
            plt.savefig(output + "_RadialVelocity.png")
            plt.close()

            plt.figure()
            vmax = np.max(np.abs(rMom_map)) / 2.0
            vmin = -vmax
            plt.imshow(rMom_map, vmin=vmin, vmax=vmax, cmap="seismic")
            plt.colorbar(label="Radial Momentum (Msun km/s)")
            plt.savefig(output + "_RadialMomentum.png")
            plt.close()

            plt.figure()
            vmax = 150
            vmin = -vmax
            plt.imshow(
                np.divide(sMom_map, mass_map), vmin=vmin, vmax=vmax, cmap="seismic"
            )
            plt.colorbar(label="Cylindrical Radial Velocity (km/s)")
            plt.savefig(output + "_CylRadialVelocity.png")
            plt.close()

            plt.figure()
            vmax = 400
            vmin = 0
            plt.imshow(
                np.divide(rotMom_map, mass_map), vmin=vmin, vmax=vmax, cmap="inferno"
            )
            plt.colorbar(label="Rotational Velocity (km/s)")
            plt.savefig(output + "_RotationalVelocity.png")
            plt.close()

    logger.info("Finished %s in %.1f s", os.path.basename(output), time.time() - t1)
