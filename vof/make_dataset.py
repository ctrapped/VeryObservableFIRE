"""Convert a FIRE snapshot into an image dataset usable with CoNNGaFit.

Depending on the options, first generates annotation files (projected mass flux, mass,
and/or rotational velocity maps), then generates the synthetic images corresponding to
those projection maps.

Written by Cameron Trapp (ctrapped@gmail.com).
"""

import logging
import os

import numpy as np

from vof.binfire.binfire import load_gas, run_binfire
from vof.generate_synthetic_image import generate_synthetic_image
from vof.io.load_data import (
    load_data_for_image_gen,
    load_masked_data_for_image_gen,
    read_stats,
)
from vof.orient_galaxy import orient_galaxy, orientation_vectors
from vof.write_annotations import write_annotations

logger = logging.getLogger(__name__)


def particles_for_binfire(particles):
    """Return a shallow copy of particles with its own copy of the velocities.

    run_binfire subtracts vel_center from particles["v"] in place, so repeated calls
    can't share it.
    """
    shielded = dict(particles)
    shielded["v"] = np.copy(particles["v"])
    return shielded


def mask_particle_data(particles, pos_center, rMax, extraArrays=()):
    """Return the particles (and extra per-particle arrays) within rMax of pos_center.

    Used to trim the larger region loaded for the shrinking-sphere centering.
    """
    N = np.shape(particles["p"])[0]
    mask = np.linalg.norm(particles["p"] - pos_center, axis=1) < rMax
    logger.debug(
        "Trimming gas particles to r < %s kpc: %d -> %d",
        rMax,
        N,
        np.count_nonzero(mask),
    )

    trimmed = {}
    for key, value in particles.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and np.shape(value)[0] == N:
            trimmed[key] = value[mask]
        else:
            trimmed[key] = value  # header and other non-per-particle entries

    trimmedExtras = []
    for arr in extraArrays:
        if np.shape(arr)[0] != N:
            raise ValueError(
                f"Per-particle array has length {np.shape(arr)[0]}, expected {N}; cannot apply radial trim consistently."
            )
        trimmedExtras.append(arr[mask])

    return trimmed, trimmedExtras


def make_dataset(
    config, fileDir, statsDir, Nsnap, output, galName, writeRadialVelocity=True
):
    """Create the annotations and/or images for every orientation of one snapshot."""
    # Config values this function's own body needs. Values only needed by generate_synthetic_image/project_image
    # (e.g. observerDistance, beamSize, num_cores, ...) are read directly out of config there instead.
    maxRadius = config["maxRadius"]
    maxHeight = config["maxHeight"]
    Nsightlines1d = config["Nsightlines1d"]
    inclinations = config["inclinations"]
    position_angles = config["position_angles"]
    speciesToRun = config["speciesToRun"]
    Nspec = config["Nspec"]
    createAnnotations = config["runBinfire"]
    createImages = config["runVOF"]
    savePNG = config["savePng"]
    createMaskFromExistingStatsDir = config["createMaskFromExistingStatsDir"]
    projectGasProperties = config["projectGasProperties"]

    outputSuffix = ""
    particles = None

    if (
        createAnnotations or projectGasProperties
    ):  # Run Binfire to create binned projection maps for radial mass flux, mass, and rotational velocities. Used as annotation files in NN training
        maskCenter = None
        maskRadius = None
        if createMaskFromExistingStatsDir:
            pos_center, vel_center, Lhat, _r_0, _ = read_stats(
                statsDir + str(Nsnap).zfill(4) + ".hdf5"
            )
            maskCenter = pos_center
            maskRadius = 100
            statsDir += "masked_"
            outputSuffix = "_masked"
            logger.debug("Set mask center")

        particles = load_gas(  # Load gas particles
            fileDir + str(Nsnap),
            statsDir + str(Nsnap).zfill(4) + ".hdf5",
            Nsnap,
            [maxRadius, maxRadius, maxHeight],
            maskCenter=maskCenter,
            maskRadius=maskRadius,
        )

        if projectGasProperties:
            Gmom_r, Gmom_s, Gmom_phi = run_binfire(
                fileDir + str(Nsnap),
                statsDir + str(Nsnap).zfill(4) + ".hdf5",
                particles_for_binfire(particles),
                Nsnap,
                output,
                [maxRadius, maxRadius, maxHeight],
                [Nsightlines1d, Nsightlines1d, Nspec],
                inclination=90 - inclinations[0],
                position_angle=position_angles[0],
                maskCenter=maskCenter,
                maskRadius=maskRadius,
                project_gas_properties=projectGasProperties,
            )

        else:
            logger.info("Creating annotation files with Binfire")
            for inclination in inclinations:
                os.makedirs(output + "i" + str(inclination) + "/", exist_ok=True)
                for position_angle in position_angles:
                    (
                        binnedMass,
                        binnedRadialMassFlux,
                        binnedPhiMassFlux,
                        binnedCylRadMassFlux,
                        binnedCylRadMassFluxCurve,
                    ) = run_binfire(
                        fileDir + str(Nsnap),
                        statsDir + str(Nsnap).zfill(4) + ".hdf5",
                        particles_for_binfire(particles),
                        Nsnap,
                        output,
                        [maxRadius, maxRadius, maxHeight],
                        [Nsightlines1d, Nsightlines1d, Nspec],
                        inclination=90 - inclination,
                        position_angle=position_angle,
                        maskCenter=maskCenter,
                        maskRadius=maskRadius,
                    )

                    write_annotations(
                        output,
                        galName,
                        inclination,
                        position_angle,
                        Nsnap,
                        outputSuffix,
                        binnedMass,
                        binnedRadialMassFlux,
                        binnedPhiMassFlux,
                        binnedCylRadMassFlux,
                        binnedCylRadMassFluxCurve,
                        writeRadialVelocity=writeRadialVelocity,
                        savePNG=savePNG,
                    )

    if createImages:
        logger.info("Creating synthetic images")

        Nsnapstring = str(Nsnap)
        pos_center, vel_center, Lhat, r0, _ = read_stats(
            statsDir + Nsnapstring.zfill(4) + ".hdf5"
        )
        snapdir = fileDir + Nsnapstring  # Directory containing the actual snapshots

        logger.debug("Snapshot directory: %s", snapdir)

        if particles is None:
            # Not already loaded for Binfire above. The stats file exists by now (read just above),
            # so load_gas only loads the region around the galaxy.
            particles = load_gas(
                snapdir,
                statsDir + Nsnapstring.zfill(4) + ".hdf5",
                Nsnap,
                [maxRadius, maxRadius, maxHeight],
            )

        # load_gas loads a box around the galaxy (a ~5 Mpc one if it had to find the center first); trim to the imaged radius
        if projectGasProperties:
            particles, (Gmom_r, Gmom_s, Gmom_phi) = mask_particle_data(
                particles,
                pos_center,
                maxRadius * 1.5,
                extraArrays=(Gmom_r, Gmom_s, Gmom_phi),
            )
        else:
            particles, _ = mask_particle_data(particles, pos_center, maxRadius * 1.5)

        gPos, gKernel, gVel = load_data_for_image_gen(particles, pos_center, vel_center)
        gMas, gTemp, speciesMassFrac = load_masked_data_for_image_gen(
            particles, gKernel, speciesToRun
        )

        for inclination in inclinations:
            os.makedirs(output + "i" + str(inclination) + "/", exist_ok=True)
            for position_angle in position_angles:
                logger.info(
                    "Generating image for i=%s, PA=%s", inclination, position_angle
                )
                image_name = (
                    output
                    + "i"
                    + str(inclination)
                    + "/"
                    + galName
                    + "_i"
                    + str(inclination)
                    + "_pa"
                    + str(position_angle)
                    + "_"
                    + str(Nsnap)
                    + "_image"
                    + outputSuffix
                )
                if os.path.isfile(image_name + "_fullSpectra.hdf5"):
                    logger.warning(
                        "Image for i=%s, PA=%s already exists, overwriting",
                        inclination,
                        position_angle,
                    )

                Lhat_rot, r0_rot = orientation_vectors(
                    Lhat, r0, inclination, position_angle
                )
                gPos_rot, gVel_rot = orient_galaxy(gPos, gVel, Lhat_rot, r0_rot)

                particleData = {}
                particleData["pos"] = gPos_rot
                particleData["vel"] = gVel_rot
                # kernel/mass/temp/speciesMassFrac/rMom/sMom/rotMom are shared references, not per-iteration
                # copies: nothing downstream mutates them in place. Do not write into them.
                particleData["kernel"] = gKernel
                particleData["mass"] = gMas
                particleData["temp"] = gTemp
                particleData["speciesMassFrac"] = speciesMassFrac

                if projectGasProperties:
                    particleData["rMom"] = Gmom_r
                    particleData["sMom"] = Gmom_s
                    particleData["rotMom"] = Gmom_phi

                generate_synthetic_image(
                    config,
                    fileDir,
                    statsDir,
                    Nsnap,
                    image_name,
                    Lhat_rot,
                    r0_rot,
                    particleData=particleData,
                )

                if config["onlySaveFirstPng"]:
                    config["savePng"] = False
