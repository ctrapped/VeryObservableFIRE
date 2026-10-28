"""Loading and preparing gas for imaging: stats files, temperatures, HI fractions, and trimming."""

import numpy as np
import pytest

from tests.synthetic_galaxy import SNAPSHOT
from vof.constants import proton_mass, unit_V
from vof.io.load_data import (
    calc_temps,
    load_data,
    load_masked_data_for_image_gen,
    read_stats,
)
from vof.make_dataset import mask_particle_data

BOLTZMANN_CGS = 1.38064852e-16  # erg/K


def test_read_stats(galaxy):
    """read_stats returns the center, velocity, orientation, and extent written to the stats file."""
    pos_center, vel_center, Lhat, r0, maxima = read_stats(galaxy.stats_file)
    np.testing.assert_array_equal(pos_center, galaxy.center)
    np.testing.assert_array_equal(vel_center, galaxy.vel_center)
    np.testing.assert_array_equal(Lhat, galaxy.Lhat)
    np.testing.assert_array_equal(r0, galaxy.r0)
    np.testing.assert_array_equal(maxima, [15.0, 15.0, 5.0])


def test_calc_temps_matches_ideal_gas():
    """T = mu m_p (gamma - 1) u / k_B, with mu set by the helium fraction and electron abundance."""
    u = np.array([50.0, 100.0])  # (km/s)^2
    ne = np.array([0.0, 1.0])
    Z = np.array([[0.02, 0.25], [0.02, 0.25]])
    y = 0.25 / (4 * 0.75)
    mu = (1 + 4 * y) / (1 + y + ne)
    m_p_grams = proton_mass * 1.98855e43
    expected = mu * m_p_grams * (2 / 3) * u * unit_V**2 / BOLTZMANN_CGS
    np.testing.assert_allclose(calc_temps(u, ne, Z), expected, rtol=1e-6)


def test_load_data_centers_and_truncates(galaxy):
    """load_data returns gas within the box around the center, shifted into the galaxy's frame."""
    pos, vel, mass, _dens, _Z, _nh, _hsml, temp = load_data(
        galaxy.snapdir, str(SNAPSHOT), 0, 10.0, galaxy.center, galaxy.vel_center
    )
    assert np.all(np.abs(pos) < 10.0)
    gas = galaxy.particles[0]
    inside = np.all(np.abs(gas["p"] - galaxy.center) < 10.0, axis=1)
    np.testing.assert_allclose(pos, gas["p"][inside] - galaxy.center, atol=1e-10)
    np.testing.assert_allclose(vel, gas["v"][inside] - galaxy.vel_center, atol=1e-10)
    assert len(mass) == len(temp) == np.count_nonzero(inside)


def gas_particles(n=50):
    """Return a minimal gas-particle dict like load_gas produces."""
    rng = np.random.default_rng(1)
    return {
        "p": rng.uniform(-20, 20, (n, 3)),
        "m": np.full(n, 1e-4),
        "h": np.full(n, 0.2),
        "rho": np.full(n, 0.05),
        "nh": np.full(n, 0.95),
        "u": np.full(n, 50.0),
        "ne": np.full(n, 0.05),
        "z": np.tile([0.02, 0.27], (n, 1)),
        "header": {"time": 0.9},
    }


def test_hi_fraction_is_physical():
    """The HI mass fraction is between 0 and the hydrogen mass fraction."""
    particles = gas_particles()
    _mass, _temp, f_HI = load_masked_data_for_image_gen(
        particles, particles["h"], "HI_21cm"
    )
    assert np.all(f_HI >= 0)
    assert np.all(f_HI <= 1 - 0.02 - 0.27)


@pytest.mark.xfail(
    strict=True,
    raises=UnboundLocalError,
    reason="Known bug: only HI_21cm sets the species mass fraction; other lines should raise a clear error",
)
def test_unsupported_species_raises_clear_error():
    """Requesting a line that isn't implemented for imaging raises ValueError."""
    particles = gas_particles()
    with pytest.raises(ValueError):
        load_masked_data_for_image_gen(particles, particles["h"], "h_alpha")


def test_mask_particle_data_trims_per_particle_arrays():
    """Per-particle arrays (and extras) are trimmed to the sphere; other entries pass through unchanged."""
    particles = gas_particles()
    extra = np.arange(len(particles["m"]))
    trimmed, (trimmed_extra,) = mask_particle_data(
        particles, np.zeros(3), 15.0, extraArrays=(extra,)
    )

    inside = np.linalg.norm(particles["p"], axis=1) < 15.0
    for key in ["p", "m", "z"]:
        np.testing.assert_array_equal(trimmed[key], particles[key][inside])
    np.testing.assert_array_equal(trimmed_extra, extra[inside])
    assert trimmed["header"] is particles["header"]


def test_mask_particle_data_rejects_mismatched_extra_array():
    """An extra array of the wrong length raises instead of being trimmed inconsistently."""
    with pytest.raises(ValueError, match="expected 50"):
        mask_particle_data(
            gas_particles(), np.zeros(3), 15.0, extraArrays=(np.zeros(10),)
        )
