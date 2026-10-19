"""FITS export: data layout and world-coordinate header values (needs the [fits] extras)."""

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")
pytest.importorskip("unyt")

# only importable with the [fits] extras
from vof.convert_to_fits import convert_to_fits


@pytest.fixture
def template(tmp_path):
    """A minimal FITS file to use as the header template."""
    path = tmp_path / "template.fits"
    fits.PrimaryHDU(np.zeros((1, 2, 2, 2))).writeto(path)
    return path


def test_cube_layout_and_header(tmp_path, template):
    """The cube is stored as (Stokes, channel, y, x) and the pixel/channel scales match the setup."""
    nx = ny = 6  # VOF images are always square
    nspec = 5
    spectra = np.random.default_rng(0).random((nx, ny, nspec))
    out = tmp_path / "fits" / "cube.fits"
    convert_to_fits(spectra, str(out), 30.0, 10000.0, 6.0, 5.2, str(template))

    with fits.open(out) as hdul:
        data, header = hdul[0].data, hdul[0].header
    assert data.shape == (1, nspec, ny, nx)
    for k in range(nspec):
        np.testing.assert_array_equal(data[0, k], spectra[:, :, k])

    fov_deg = np.degrees(30.0 / 10000.0)  # 30 kpc seen from 10 Mpc
    assert header["CDELT1"] == pytest.approx(-fov_deg / nx, rel=1e-6)
    assert header["CDELT2"] == pytest.approx(fov_deg / ny, rel=1e-6)
    assert header["CDELT3"] == pytest.approx(-5200.0)  # m/s per channel
    assert header["BMAJ"] == pytest.approx(6.0 / 3600)
