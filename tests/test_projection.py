"""Image formation: smoothing kernel, optically thin deposition, and per-sightline spectra."""

import numpy as np
import pytest

from vof import project_image
from vof.calc_sightline_spectrum import calc_sightline_spectrum
from vof.constants import c_km_s, h
from vof.emission_species import get_emission_species_parameters
from vof.project_image import deposit_particles, sphere_kernel

N_PIX = 16
MAX_R = 8.0  # kpc, so pixels are 1 kpc
DISTANCE = 10000.0  # kpc
N_SPEC = 16
BANDWIDTH_KM_S = 320.0


def hi_bandwidth_hz():
    """Return the HI 21cm bandwidth in Hz, computed as very_observable_fire.main does."""
    E_upper, E_lower = get_emission_species_parameters("HI_21cm")[3:5]
    f0 = (E_upper - E_lower) / h
    return (
        f0
        * c_km_s
        * (1 / (c_km_s - BANDWIDTH_KM_S / 2) - 1 / (c_km_s + BANDWIDTH_KM_S / 2))
    )


def deposit(y, z, kernel, mass, **kwargs):
    """Deposit HI particles at image coordinates (y, z) kpc; return deposit_particles' output."""
    # allow the multi-pixel kernels used here
    kwargs.setdefault("minKernelSanityCheck", 10.0)
    n = len(mass)
    # already in the observer's frame: x is the distance along the line of sight
    pos = np.stack([np.full(n, DISTANCE), y, z], axis=1)
    return deposit_particles(
        pos,
        np.zeros((n, 3)),
        np.asarray(kernel, float),
        np.asarray(mass, float),
        np.full(n, 5000.0),
        np.full(n, 0.7),
        N_PIX,
        2 * MAX_R / N_PIX / DISTANCE,
        "HI_21cm",
        N_SPEC,
        hi_bandwidth_hz(),
        DISTANCE,
        MAX_R,
        **kwargs,
    )


@pytest.mark.parametrize("d", [0, 1, 3, 6])
def test_sphere_kernel_shape(d):
    """The kernel is (2d+1) square, symmetric, peaks at 1 in the center, and is 0 at the corners."""
    k = sphere_kernel(d)
    assert k.shape == (2 * d + 1, 2 * d + 1)
    np.testing.assert_array_equal(k, k.T)
    np.testing.assert_array_equal(k, k[::-1, ::-1])
    assert k[d, d] == 1.0
    assert k.max() == 1.0
    if d > 0:
        assert k[0, 0] == 0.0


@pytest.fixture
def fft_calls(monkeypatch):
    """Count calls to the FFT used for large particle groups, to check which path a test exercised."""
    calls = []
    real_rfft2 = project_image.rfft2

    def counting_rfft2(*args, **kwargs):
        calls.append(1)
        return real_rfft2(*args, **kwargs)

    monkeypatch.setattr(project_image, "rfft2", counting_rfft2)
    return calls


def test_deposited_mass_is_conserved(fft_calls):
    """Projected mass maps sum to the total mass of particles well inside the image."""
    rng = np.random.default_rng(0)
    n = 600
    y, z = rng.uniform(-4, 4, (2, n))
    kernel = rng.choice([0.4, 1.6, 2.6], n)  # kernel radii of 0, 1, and 2 pixels
    mass = rng.uniform(1, 2, n)
    zeros = np.zeros(n)
    _emission, mass_map, *_ = deposit(
        y,
        z,
        kernel,
        mass,
        projectGasProperties=True,
        gRmom=zeros,
        gSmom=zeros,
        gRotMom=zeros,
    )
    assert mass_map.sum() == pytest.approx(mass.sum(), rel=1e-10)
    assert fft_calls, "expected the 2-pixel group to be large enough for the FFT path"


def test_fft_path_matches_direct_stamping(fft_calls):
    """Large groups (FFT convolution) give the same image as small groups (direct stamping)."""
    one = deposit([0.3], [-1.2], [2.6], [1.0])
    assert not fft_calls
    many = deposit(
        np.full(300, 0.3), np.full(300, -1.2), np.full(300, 2.6), np.ones(300)
    )
    assert fft_calls
    assert one.sum() > 0
    np.testing.assert_allclose(many / 300, one, rtol=1e-8, atol=1e-10 * one.max())


def test_deposit_rejects_badly_centered_data():
    """Particles that all fall outside the image raise instead of producing an empty cube."""
    with pytest.raises(ValueError, match="centering"):
        deposit([50.0], [50.0], [0.5], [1.0])


def test_unknown_species_raises():
    """Asking for an emission line that isn't defined raises ValueError."""
    with pytest.raises(ValueError, match="not defined"):
        get_emission_species_parameters("OIII_5007")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.xfail(
    strict=True,
    reason="Known bug: a sightline that hits no particles normalizes 0/0 -> NaN, which smoothing spreads over the cube",
)
def test_empty_sightline_gives_zero_spectrum():
    """A sightline that misses every particle has a zero (not NaN) spectrum."""
    # observer-centered; the sightline below points along +x, 7 kpc from the particle
    pos = np.array([[DISTANCE, 5.0, 5.0]])
    _spectrum, emission = calc_sightline_spectrum(
        np.array([1.0, 0.0, 0.0]),
        pos,
        np.zeros((1, 3)),
        np.array([0.5]),
        np.array([1.0]),
        np.array([5000.0]),
        np.array([0.7]),
        2 * MAX_R / N_PIX / DISTANCE,
        "HI_21cm",
        N_SPEC,
        hi_bandwidth_hz(),
        DISTANCE,
        calcThermalLevels=False,
    )
    np.testing.assert_array_equal(emission, np.zeros(N_SPEC))
