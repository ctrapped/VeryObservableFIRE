"""Run the full ``vof`` command on the synthetic galaxy and compare to stored reference ("golden") outputs.

The golden files in tests/data/ pin the pipeline's current output. A difference means the
science output changed: if that was intended (e.g. a physics fix), regenerate them with
``pytest tests/test_end_to_end.py --update-golden`` and commit the new files with the fix.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from vof import very_observable_fire

GOLDEN_DIR = Path(__file__).parent / "data"

CASES = {
    # Fast optically thin projection, two position angles
    "thin": {},
    # Also deposit Binfire's per-particle gas properties into annotation maps
    "gas_properties": {"projectGasProperties": True, "position_angles": [30]},
    # Full per-sightline spectral synthesis (slower, so fewer pixels)
    "spectral": {
        "runAsOpticallyThinProjection": False,
        "Nsightlines1d": 8,
        "position_angles": [0],
    },
}


def run_vof(param_file, monkeypatch):
    """Run the vof command-line entry point on param_file with a fixed noise seed."""
    np.random.seed(0)
    monkeypatch.setattr(sys, "argv", ["vof", str(param_file)])
    very_observable_fire.main()


def read_outputs(output_dir):
    """Return {"<file stem>/<dataset>": array} for every datacube VOF wrote."""
    outputs = {}
    for path in sorted(Path(output_dir).rglob("*_fullSpectra.hdf5")):
        with h5py.File(path) as f:
            for name, dataset in f.items():
                outputs[f"{path.stem}/{name}"] = dataset[()]
    return outputs


@pytest.mark.parametrize("case", CASES)
def test_matches_golden_output(case, galaxy, write_param_file, monkeypatch, request):
    """The datacubes and annotation maps match the stored reference outputs."""
    param_file = write_param_file(galaxy, **CASES[case])
    run_vof(param_file, monkeypatch)
    outputs = read_outputs(param_file.parent / "output")

    n_orientations = len(CASES[case].get("position_angles", [0, 90]))
    assert len([k for k in outputs if k.endswith("/spectra")]) == n_orientations
    for name, array in outputs.items():
        assert np.all(np.isfinite(array)), f"{name} has non-finite values"

    golden_file = GOLDEN_DIR / f"golden_{case}.npz"
    if request.config.getoption("--update-golden"):
        GOLDEN_DIR.mkdir(exist_ok=True)
        np.savez_compressed(
            golden_file, **{k: v.astype(np.float32) for k, v in outputs.items()}
        )
        pytest.skip(f"Wrote {golden_file.name}")

    golden = np.load(golden_file)
    assert sorted(golden.files) == sorted(outputs), (
        "VOF wrote different datasets than the golden output"
    )
    for name in golden.files:
        actual, expected = outputs[name], golden[name]
        assert actual.shape == expected.shape, name
        if name.endswith("velocity_annotation"):
            # velocity = momentum / mass, which is round-off noise in (nearly) empty pixels; skip those
            mass = golden[name.rsplit("/", 1)[0] + "/mass_annotation"]
            filled = mass > 1e-6 * mass.max()
            actual, expected = actual[filled], expected[filled]
        # float32 storage; tolerance is relative to each map's peak so near-zero pixels don't dominate
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-5 * np.max(np.abs(expected)),
            err_msg=name,
        )


def test_image_attributes_and_pngs(galaxy, write_param_file, monkeypatch):
    """Each datacube records the observation setup, and only the first orientation gets PNG previews."""
    param_file = write_param_file(galaxy, savePng=True, onlySaveFirstPng=True)
    run_vof(param_file, monkeypatch)
    out = param_file.parent / "output" / "i60"

    with h5py.File(out / "synthetic_i60_pa0_600_image_fullSpectra.hdf5") as f:
        assert f["spectra"].shape == (24, 24, 24)
        assert f.attrs["fov_kpc"] == 30.0
        assert f.attrs["observer_distance_kpc"] == 10000
        assert f.attrs["beam_arcsec"] == pytest.approx(30)
        assert f.attrs["dnu_kmps"] == pytest.approx(20.0)

    assert (out / "synthetic_i60_pa0_600_image_ZerothMomentMap.png").exists()
    assert not (out / "synthetic_i60_pa90_600_image_ZerothMomentMap.png").exists()
