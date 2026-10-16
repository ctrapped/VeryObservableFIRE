import argparse
import logging
import time

import numpy as np

from vof.constants import (
    arcsec,
    c_km_s,
    h,
)
from vof.emission_species import get_emission_species_parameters
from vof.load_param_file import load_params
from vof.logging_setup import add_logging_arguments, configure_logging
from vof.make_dataset import make_dataset

logger = logging.getLogger(__name__)

####Runs VeryObservableFIRE to create synthetic images from given observational parameters.
####Also creates corresponding projected and deprojected maps of radial mass flux, rotational velocity, and mass for the purposes of neural network training
####To use, copy configs/param_template.param, modify it, and pass the path to it as the first argument.
####    e.g. vof configs/param_template.param   (or: python -m vof configs/param_template.param)
####Optional overrides (any order): --galName, --minSnap, --maxSnap, --inclination (see --help)
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023


def parse_args():
    parser = argparse.ArgumentParser(
        prog="vof",
        description="Run VeryObservableFIRE to generate synthetic images/annotations from a parameter file.",
    )
    parser.add_argument(
        "paramFile", help="Path to the .param file (e.g. configs/param_template.param)"
    )
    parser.add_argument(
        "--galName", default=None, help="Override galName from the parameter file"
    )
    parser.add_argument(
        "--minSnap",
        type=int,
        default=None,
        help="Override the starting snapshot number",
    )
    parser.add_argument(
        "--maxSnap",
        type=int,
        default=None,
        help="Override the ending snapshot number (defaults to minSnap if omitted)",
    )
    parser.add_argument(
        "--inclination",
        type=int,
        default=None,
        help="Restrict to a single inclination, overriding the inclinations list in the parameter file",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def main():
    startTime = time.time()
    args = parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)
    if args.maxSnap is None:
        args.maxSnap = args.minSnap

    # `config` is passed down through make_dataset/generate_synthetic_image/project_image as a single dict;
    # each of those functions only takes the (snapshot, inclination, position_angle, ...) identity/loop
    # arguments explicitly and reads everything else out of config.
    config = load_params(
        args.paramFile,
        galName=args.galName,
        minSnap=args.minSnap,
        maxSnap=args.maxSnap,
        inclination=args.inclination,
    )

    galName = config["galName"]
    minSnap = config["minSnap"]
    maxSnap = config["maxSnap"]
    snapStep = config["snapStep"]
    fileDir = config["fileDir"]
    statsDir = config["statsDir"]
    output = config["output"]

    logger.info("Snapshot directory: %s", fileDir)

    config["targetBeamSize"] = (
        config["targetBeamSize"] * arcsec
    )  # Convert from arcseconds (as given in the param file) to radians
    config["bandwidth_km_s"] = config["Nchannels"] * config["res_km_s"]
    config["beamSize"] = (
        2 * config["maxRadius"] / config["Nsightlines1d"] / config["observerDistance"]
    )

    (
        _mass_species,
        _g_upper,
        _g_lower,
        E_upper,
        E_lower,
        _A_ul,
        _gamma_ul,
        _Glevels,
        _Elevels,
        _n_u_fraction,
        _n_l_fraction,
    ) = get_emission_species_parameters(config["speciesToRun"])
    f0 = (E_upper - E_lower) / h  # in Hz
    config["Nspec"] = int(np.ceil(config["bandwidth_km_s"] / config["res_km_s"]))
    config["bandwidth"] = (
        f0
        * c_km_s
        * (
            1 / (c_km_s - config["bandwidth_km_s"] / 2)
            - 1 / (c_km_s + config["bandwidth_km_s"] / 2)
        )
    )

    logger.info(
        "Observation: beam size %g rad, f0 %g Hz, bandwidth %g Hz",
        config["beamSize"],
        f0,
        config["bandwidth"],
    )

    initialSavePng = config[
        "savePng"
    ]  # make_dataset may turn this off partway through a snapshot (onlySaveFirstPng); restore it for the next one
    for Nsnap in range(minSnap, maxSnap + 1, snapStep):
        logger.info("Processing snapshot %d", Nsnap)
        make_dataset(config, fileDir, statsDir, Nsnap, output, galName)
        config["savePng"] = initialSavePng

    logger.info("Finished all snapshots in %.1f s", time.time() - startTime)


if __name__ == "__main__":
    main()
