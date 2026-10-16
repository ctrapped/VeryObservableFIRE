import logging

####Shared logging configuration for the command-line entry points (vof, vof-to-fits, vof-augment).
####Library modules only create a logger (logging.getLogger(__name__)) and never configure output themselves.

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def add_logging_arguments(parser):
    # Add the -v/--verbose and -q/--quiet flags to an argparse parser
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show debug messages (per-step details, timings, intermediate values)",
    )
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show warnings and errors",
    )


def configure_logging(verbose=False, quiet=False):
    # Third-party libraries (matplotlib, h5py, ...) stay at INFO/WARNING; -v only turns on debug output from vof itself
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )
    logging.getLogger("vof").setLevel(level)
