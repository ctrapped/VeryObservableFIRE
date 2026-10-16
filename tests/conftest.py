"""Shared pytest fixtures: the synthetic galaxy and parameter files for running VOF on it."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # tests never open plot windows

import pytest

from tests.synthetic_galaxy import SNAPSHOT, write_stats_file, write_synthetic_snapshot

MAX_RADIUS = 15.0
MAX_HEIGHT = 5.0

# A small, fast observation of the synthetic galaxy. Tests override individual values.
BASE_PARAMS = {
    "galName": "synthetic",
    "minSnap": SNAPSHOT,
    "maxSnap": SNAPSHOT,
    "snapStep": 1,
    "observerDistance": 10000,
    "observerVelocity": [0, 0, 0],
    "maxRadius": MAX_RADIUS,
    "maxHeight": MAX_HEIGHT,
    "targetBeamSize": 30,
    "Nsightlines1d": 24,
    "phiObs": 0,
    "inclinations": [60],
    "position_angles": [0, 90],
    "speciesToRun": "HI_21cm",
    "res_km_s": 20.0,
    "Nchannels": 24,
    "noiseAmplitude": 1e18,
    "runBinfire": False,
    "runVOF": True,
    "savePng": False,
    "onlySaveFirstPng": True,
    "createMaskFromExistingStatsDir": False,
    "num_cores": 1,
    "projectGasProperties": False,
    "runAsOpticallyThinProjection": True,
}


def pytest_addoption(parser):
    """Add --update-golden, to rewrite the stored end-to-end reference outputs."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="Overwrite tests/data/golden_*.npz with the current outputs instead of comparing to them",
    )


@pytest.fixture(scope="session")
def galaxy(tmp_path_factory):
    """The synthetic galaxy, written once per test session, with a stats file of its true orientation.

    Treat it as read-only; use fresh_galaxy for tests that write next to the snapshot.
    """
    root = tmp_path_factory.mktemp("galaxy")
    truth = write_synthetic_snapshot(root / f"snapdir_{SNAPSHOT}")
    write_stats_file(truth, MAX_RADIUS, MAX_HEIGHT)
    return truth


@pytest.fixture
def fresh_galaxy(tmp_path):
    """A newly written synthetic galaxy with no stats file, so VOF has to find its center."""
    return write_synthetic_snapshot(tmp_path / f"snapdir_{SNAPSHOT}")


@pytest.fixture
def write_param_file(tmp_path):
    """Return a function that writes a .param file for a galaxy (with overrides) and returns its path."""

    def _write(galaxy, **overrides):
        params = dict(BASE_PARAMS)
        # VOF appends the snapshot number to fileDir and "<snapshot>.hdf5" to statsDir
        params["fileDir"] = galaxy.snapdir[: -len(str(SNAPSHOT))]
        params["statsDir"] = galaxy.stats_file[: -len(f"{SNAPSHOT:04d}.hdf5")]
        params["output"] = str(tmp_path / "output") + "/"
        params.update(overrides)
        path = tmp_path / "test.param"
        path.write_text("".join(f"{key} = {value}\n" for key, value in params.items()))
        return path

    return _write
