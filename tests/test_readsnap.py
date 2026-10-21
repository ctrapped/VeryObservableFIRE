"""Snapshot readers: files are found in each supported layout, and values match what was written, in physical units."""

import numpy as np
import pytest

from tests.synthetic_galaxy import N_METAL_FIELDS, SNAPSHOT, write_synthetic_snapshot
from vof.io.readsnap_vof import (
    check_if_filename_exists,
    readsnap_initial,
    readsnap_trunc,
)

READ_OPTIONS = {
    "snapshot_name": "snapshot",
    "extension": ".hdf5",
    "h0": 1,
    "cosmological": 1,
}


def read_initial(galaxy, ptype, **kwargs):
    """Call readsnap_initial on the synthetic galaxy with VOF's usual options."""
    return readsnap_initial(
        galaxy.snapdir, str(SNAPSHOT), ptype, **READ_OPTIONS, **kwargs
    )


def read_trunc(galaxy, ptype, mask, **kwargs):
    """Call readsnap_trunc on the synthetic galaxy with VOF's usual options."""
    return readsnap_trunc(
        galaxy.snapdir, str(SNAPSHOT), ptype, mask, **READ_OPTIONS, **kwargs
    )


@pytest.mark.parametrize("ptype", [0, 1, 4])
def test_positions_are_physical_kpc(galaxy, ptype):
    """Positions are converted from comoving kpc/h and concatenated across both files in order."""
    result = read_initial(galaxy, ptype)
    np.testing.assert_allclose(result["p"], galaxy.particles[ptype]["p"], rtol=1e-12)


def test_density_only_loaded_on_request(galaxy):
    """readsnap_initial reads gas densities only with load_density=True."""
    assert "rho" not in read_initial(galaxy, 0)
    rho = read_initial(galaxy, 0, load_density=True)["rho"]
    np.testing.assert_allclose(rho, galaxy.particles[0]["rho"], rtol=1e-12)


@pytest.fixture
def gas_mask(galaxy):
    """A boolean mask selecting a random ~60% of the gas particles."""
    return np.random.default_rng(0).random(len(galaxy.particles[0]["m"])) < 0.6


def test_truncated_gas_fields(galaxy, gas_mask):
    """readsnap_trunc returns only the masked particles, with every gas field in physical units."""
    result = read_trunc(galaxy, 0, gas_mask)
    truth = {key: value[gas_mask] for key, value in galaxy.particles[0].items()}
    for key in ["v", "m", "rho", "h", "u", "ne", "nh"]:
        np.testing.assert_allclose(result[key], truth[key], rtol=1e-12, err_msg=key)
    # Only total metallicity and helium are read by default
    np.testing.assert_allclose(result["z"], truth["z"][:, :2], rtol=1e-12)


def test_truncated_gas_options(galaxy, gas_mask):
    """The load_density/load_ids/n_metal_fields options skip or add fields as documented."""
    full = read_trunc(galaxy, 0, gas_mask)
    assert {"rho", "pid", "cid", "gen"} <= set(full)

    slim = read_trunc(galaxy, 0, gas_mask, load_density=False, load_ids=False)
    assert not {"rho", "pid", "cid", "gen"} & set(slim)

    all_metals = read_trunc(galaxy, 0, gas_mask, n_metal_fields=None)
    assert all_metals["z"].shape == (np.count_nonzero(gas_mask), N_METAL_FIELDS)


def test_dark_matter_velocities_are_loaded(galaxy):
    """DM velocities are needed for the galaxy's center-of-mass velocity (regression test)."""
    mask = np.ones(len(galaxy.particles[1]["m"]), dtype=bool)
    result = read_trunc(galaxy, 1, mask)
    np.testing.assert_allclose(result["v"], galaxy.particles[1]["v"], rtol=1e-12)


def test_star_ages(galaxy):
    """Star formation times are read for stars."""
    mask = np.ones(len(galaxy.particles[4]["m"]), dtype=bool)
    np.testing.assert_allclose(
        read_trunc(galaxy, 4, mask)["age"], galaxy.particles[4]["age"]
    )


@pytest.mark.parametrize("reader", ["initial", "trunc"])
def test_missing_snapshot_raises(tmp_path, reader):
    """A directory with no snapshot raises FileNotFoundError naming the directory."""
    with pytest.raises(FileNotFoundError, match="No snapshot 600 found"):
        if reader == "initial":
            readsnap_initial(str(tmp_path), "600", 0, **READ_OPTIONS)
        else:
            readsnap_trunc(str(tmp_path), "600", 0, np.ones(1, bool), **READ_OPTIONS)


@pytest.mark.parametrize(
    "layout",
    [
        "snapshot_600.hdf5",
        "snap_600.hdf5",
        "snapshot_600.0.hdf5",
        "snapdir_600/snapshot_600.0.hdf5",
        "snapdir_600/snap_600.0.hdf5",
    ],
)
def test_finds_snapshot_layouts(tmp_path, layout):
    """fileDir is the folder holding the snapshots; single files and snapdir_XXX/ subfolders are both found."""
    path = tmp_path / layout
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    fname, _, extension = check_if_filename_exists(str(tmp_path), "600")
    assert fname == f"{tmp_path}/{layout}"
    assert extension == ".hdf5"


def test_reads_single_file_snapshot(tmp_path):
    """A one-file snapshot (fileDir/snapshot_600.hdf5) reads the same particles as the 2-file one."""
    galaxy = write_synthetic_snapshot(tmp_path, single_file=True)
    np.testing.assert_allclose(
        read_initial(galaxy, 0)["p"], galaxy.particles[0]["p"], rtol=1e-12
    )
