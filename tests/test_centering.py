"""Binfire: finding a galaxy's center, bulk velocity, and spin axis from scratch."""

import numpy as np
import pytest

from tests.synthetic_galaxy import SNAPSHOT
from vof.binfire.binfire import load_gas, run_binfire
from vof.io.load_data import read_stats
from vof.make_dataset import particles_for_binfire

MAXIMA = [15.0, 15.0, 5.0]
NBINS = [16, 16, 8]


@pytest.fixture
def centered_galaxy(fresh_galaxy, tmp_path):
    """Run Binfire on a galaxy with no stats file; return (truth, binned maps)."""
    particles = load_gas(
        fresh_galaxy.snapdir, fresh_galaxy.stats_file, SNAPSHOT, MAXIMA
    )
    maps = run_binfire(
        fresh_galaxy.snapdir,
        fresh_galaxy.stats_file,
        particles_for_binfire(particles),
        SNAPSHOT,
        str(tmp_path / "out") + "/",
        MAXIMA,
        NBINS,
        inclination=90,
        position_angle=0,
    )
    return fresh_galaxy, maps


def test_finds_center_velocity_and_spin_axis(centered_galaxy):
    """The stats file Binfire writes recovers the galaxy's true center, velocity, and orientation."""
    truth, _maps = centered_galaxy
    pos_center, vel_center, Lhat, r0, _maxima = read_stats(truth.stats_file)

    assert np.linalg.norm(pos_center - truth.center) < 0.1  # kpc
    assert np.linalg.norm(vel_center - truth.vel_center) < 1.0  # km/s
    assert np.degrees(np.arccos(np.dot(Lhat, truth.Lhat))) < 1.0
    # r0 lies in the disk plane
    assert np.dot(r0, Lhat) == pytest.approx(0.0, abs=1e-10)
