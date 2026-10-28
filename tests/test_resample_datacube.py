"""Resampling existing VOF cubes: the re-observed cube, its annotations, and the CLI."""

import sys

import h5py
import numpy as np
import pytest
import scipy

from vof.constants import arcsec
from vof.resample_datacube import downsample, main, resample_datacube

N = 16  # pixels per side of the input cube
NSPEC = 4
FACTOR = 2
FOV_KPC = 30.0
DISTANCE_KPC = 10000.0
DNU_KMPS = 20.0
PIXEL_ARCSEC = FOV_KPC / DISTANCE_KPC / arcsec / N  # about 39 arcsec
NOISE = 1e18  # cm^-2


def write_cube(path, ideal_image=None, annotations=True):
    """Write a minimal VOF output (see generate_synthetic_image.py); returns its annotation maps."""
    rng = np.random.default_rng(0)
    if ideal_image is None:
        ideal_image = rng.random((N, N, NSPEC)) * 1e20
    mass = rng.random((N, N))
    velocities = {
        name: rng.uniform(-200, 200, (N, N))
        for name in ("radial", "cylindrical_radial", "rotational")
    }
    with h5py.File(path, "w") as hf:
        hf.create_dataset("spectra", data=ideal_image)
        hf.create_dataset("ideal_image", data=ideal_image)
        hf.attrs["fov_kpc"] = FOV_KPC
        hf.attrs["observer_distance_kpc"] = DISTANCE_KPC
        hf.attrs["beam_arcsec"] = PIXEL_ARCSEC
        hf.attrs["dnu_kmps"] = DNU_KMPS
        if annotations:
            hf.create_dataset("mass_annotation", data=mass)
            for name, velocity in velocities.items():
                hf.create_dataset(f"{name}_velocity_annotation", data=velocity)
            hf.attrs["annotation_mass_units"] = "Msun"
            hf.attrs["annotation_velocity_units"] = "km/s"
    return mass, velocities


def test_resample_keeps_metadata_and_conserves_mass(tmp_path):
    """The output is FACTOR x coarser, with the new beam, the total mass, and mass-weighted velocities."""
    mass, velocities = write_cube(tmp_path / "in.hdf5")
    out = tmp_path / "resampled" / "out.hdf5"
    resample_datacube(
        str(tmp_path / "in.hdf5"), str(out), FACTOR, NOISE, 3 * PIXEL_ARCSEC
    )

    with h5py.File(out) as hf:
        assert hf["spectra"].shape == (N // FACTOR, N // FACTOR, NSPEC)
        # The field of view doesn't change, only the pixel size
        assert hf.attrs["fov_kpc"] == FOV_KPC
        assert hf.attrs["observer_distance_kpc"] == DISTANCE_KPC
        assert hf.attrs["beam_arcsec"] == pytest.approx(3 * PIXEL_ARCSEC)
        assert hf.attrs["dnu_kmps"] == DNU_KMPS
        assert hf.attrs["annotation_mass_units"] == "Msun"
        assert hf.attrs["annotation_velocity_units"] == "km/s"
        new_mass = hf["mass_annotation"][()]
        new_rotation = hf["rotational_velocity_annotation"][()]

    assert new_mass.shape == (N // FACTOR, N // FACTOR)
    assert new_mass.sum() == pytest.approx(mass.sum())
    block = np.s_[:FACTOR, :FACTOR]  # the input pixels that make up output pixel (0, 0)
    expected = np.sum(mass[block] * velocities["rotational"][block]) / np.sum(
        mass[block]
    )
    assert new_rotation[0, 0] == pytest.approx(expected)


def test_resample_smooths_to_the_new_beam(tmp_path):
    """A point source is spread by a Gaussian beam of the requested FWHM, then block-averaged."""
    ideal = np.zeros((N, N, NSPEC))
    ideal[N // 2, N // 2, :] = 1e20
    write_cube(tmp_path / "in.hdf5", ideal, annotations=False)
    beam_arcsec = 3 * PIXEL_ARCSEC
    resample_datacube(
        str(tmp_path / "in.hdf5"), str(tmp_path / "out.hdf5"), FACTOR, 1.0, beam_arcsec
    )

    sigma_pixels = 3 / (2 * np.sqrt(2 * np.log(2)))  # FWHM of 3 input pixels
    expected = downsample(
        scipy.ndimage.gaussian_filter(ideal, sigma_pixels, axes=[0, 1]), FACTOR
    )
    with h5py.File(tmp_path / "out.hdf5") as hf:
        # Noise of 1 cm^-2 is negligible next to the source
        np.testing.assert_allclose(hf["spectra"][()], expected, atol=100)


def test_resample_noise_has_target_amplitude(tmp_path):
    """On an empty cube, the output is pure noise with the requested standard deviation."""
    write_cube(tmp_path / "in.hdf5", np.zeros((N, N, NSPEC)), annotations=False)
    resample_datacube(
        str(tmp_path / "in.hdf5"),
        str(tmp_path / "out.hdf5"),
        FACTOR,
        NOISE,
        3 * PIXEL_ARCSEC,
    )
    with h5py.File(tmp_path / "out.hdf5") as hf:
        assert np.std(hf["spectra"][()]) == pytest.approx(NOISE)


def test_cli_resamples_a_directory(tmp_path, monkeypatch):
    """vof-resample DIR OUT resamples every .hdf5 cube in DIR into OUT, keeping their names, plus PNGs."""
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a_fullSpectra.hdf5", "b_fullSpectra.hdf5"):
        write_cube(images / name)
    (images / "notes.txt").write_text("not a cube")
    output = tmp_path / "resampled"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vof-resample",
            str(images),
            str(output),
            "--spatial-resampling-factor",
            str(FACTOR),
            "--target-res",
            str(3 * PIXEL_ARCSEC),
            "--target-sensitivity",
            str(NOISE),
            "--save-png",
        ],
    )
    main()

    cubes = sorted(p.name for p in output.glob("*.hdf5"))
    assert cubes == ["a_fullSpectra.hdf5", "b_fullSpectra.hdf5"]
    # --save-png writes the pipeline's diagnostic PNGs, named like the pipeline's
    assert (output / "a_ZerothMomentMap.png").exists()
    assert (output / "a_RotationalVelocity.png").exists()
    with h5py.File(output / "a_fullSpectra.hdf5") as hf:
        assert hf["spectra"].shape == (N // FACTOR, N // FACTOR, NSPEC)
