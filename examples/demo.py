"""Demo: build a small synthetic galaxy, make a synthetic image with VOF, and plot the results.

Needs no simulation data and runs in a few seconds. From the repository root:

    python examples/demo.py                 # writes examples/demo_output/demo.png
    or
    python examples/demo.py --output DIR

The galaxy is a rotating disk the test suite uses (tests/synthetic_galaxy.py), written in
the same file format as a FIRE snapshot, so VOF runs on it exactly as it would on
simulation data.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.transforms import offset_copy

REPO_ROOT = Path(__file__).resolve().parents[1]
# So the synthetic galaxy can be imported from tests/
sys.path.insert(0, str(REPO_ROOT))

from tests.synthetic_galaxy import (  # after the sys.path line above
    SNAPSHOT,
    write_stats_file,
    write_synthetic_snapshot,
)

INCLINATION = 60
MAX_RADIUS = 15.0  # kpc
N_PIXELS = 48
RES_KM_S = 10.0
N_CHANNELS = 48
NOISE = 3e18  # noise per channel, cm^-2

# The observation to simulate, in the same key = value format as configs/param_template.param
PARAMS = """\
galName = demo
minSnap = {snapshot}
maxSnap = {snapshot}
snapStep = 1
fileDir = {snapdir}/
statsDir = {stats_dir}
output = {output}/

observerDistance = 10000        # 10 Mpc
observerVelocity = [0, 0, 0]
maxRadius = {max_radius}        # image out to 15 kpc from the center
maxHeight = 5
targetBeamSize = 20             # arcseconds
Nsightlines1d = {n_pixels}
phiObs = 0
inclinations = [{inclination}]
position_angles = [0]

speciesToRun = HI_21cm
res_km_s = {res_km_s}
Nchannels = {n_channels}
noiseAmplitude = {noise}

runBinfire = False
runVOF = True
savePng = False
onlySaveFirstPng = True
createMaskFromExistingStatsDir = False
num_cores = 1
projectGasProperties = True     # also write the true gas properties, for comparison
runAsOpticallyThinProjection = True
"""


def run_vof(output):
    """Write the synthetic galaxy and a parameter file into output, then run VOF on them."""
    galaxy = write_synthetic_snapshot(output / "snapshots" / f"snapdir_{SNAPSHOT}")
    write_stats_file(galaxy, MAX_RADIUS)  # the galaxy's center and orientation

    param_file = output / "demo.param"
    param_file.write_text(
        PARAMS.format(
            snapshot=SNAPSHOT,
            snapdir=output / "snapshots",
            stats_dir=galaxy.stats_file[: -len(f"{SNAPSHOT:04d}.hdf5")],
            output=output / "vof_output",
            max_radius=MAX_RADIUS,
            n_pixels=N_PIXELS,
            inclination=INCLINATION,
            res_km_s=RES_KM_S,
            n_channels=N_CHANNELS,
            noise=NOISE,
        )
    )
    # Exactly what you would type at the command line
    subprocess.run(
        [sys.executable, "-m", "vof", str(param_file)], cwd=REPO_ROOT, check=True
    )

    return next((output / "vof_output").rglob("*_fullSpectra.hdf5"))


def plot(cube_file, figure_file):
    """Plot the mock observation next to the true gas properties it was made from."""
    with h5py.File(cube_file) as f:
        # Noisy datacube: (y, x, velocity channel), in cm^-2 per channel
        cube = f["spectra"][()]
        mass = f["mass_annotation"][()]  # Msun per pixel
        rotation = f["rotational_velocity_annotation"][()]  # km/s

    velocities = (np.arange(N_CHANNELS) - (N_CHANNELS - 1) / 2) * RES_KM_S
    moment0 = cube.sum(axis=2)  # total HI column density
    detected = moment0 > 5 * NOISE * np.sqrt(N_CHANNELS)  # 5-sigma detection
    moment1 = np.where(detected, (cube * velocities).sum(axis=2) / moment0, np.nan)
    # Pixels holding disk gas, not just diffuse halo gas
    disk = mass > 1e-3 * mass.max()

    extent = [-MAX_RADIUS, MAX_RADIUS, -MAX_RADIUS, MAX_RADIUS]
    # Each panel's colorbar label doubles as its title
    panels = [
        (
            "Mock observation\nHI column density [cm$^{-2}$]",
            np.clip(moment0, 1e19, None),
            {"norm": LogNorm(1e19, moment0.max()), "cmap": "inferno"},
        ),
        (
            "Mock observation\nLine-of-sight velocity [km/s]",
            moment1,
            {"vmin": -200, "vmax": 200, "cmap": "RdBu_r"},
        ),
        (
            "Ground truth\nGas mass per pixel [M$_\\odot$]",
            np.where(mass > 0, mass, np.nan),
            {"norm": LogNorm(), "cmap": "viridis"},
        ),
        (
            "Ground truth\nRotational velocity [km/s]",
            np.where(disk, rotation, np.nan),
            {"vmin": 0, "vmax": 250, "cmap": "magma"},
        ),
    ]

    labelsize = 16
    cbar_ticksize = 14
    ticksize = 14
    cbar_labelsize = 16
    # points between the top of each colorbar and the baseline of its label
    cbar_label_offset = 30
    cbar_label_line_spacing = 20  # points between the baselines of the two label lines
    tickmarks = [-15, -10, -5, 0, 5, 10, 15]
    ticklabels = ["-15", "-10", "-5", "0", "5", "10", "15"]

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2), constrained_layout=True)
    for ax, (label, image, style) in zip(axes, panels):
        im = ax.imshow(image, origin="lower", extent=extent, **style)
        ax.set_xlabel("x [kpc]", fontsize=labelsize)

        cbar = fig.colorbar(im, ax=ax, location="top", shrink=0.9, pad=0.02)
        cbar.ax.tick_params(labelsize=cbar_ticksize)

        for i, line in enumerate(reversed(label.split("\n"))):
            offset = cbar_label_offset + i * cbar_label_line_spacing
            above_ticks = offset_copy(
                cbar.ax.transAxes, fig=fig, y=offset, units="points"
            )
            cbar.ax.text(
                0.5,
                1,
                line,
                transform=above_ticks,
                ha="center",
                va="baseline",
                fontsize=cbar_labelsize,
            )
        ax.set_xticks(tickmarks, ticklabels, fontsize=ticksize)
        ax.set_yticks(tickmarks, ticklabels, fontsize=ticksize)
    axes[0].set_ylabel("y [kpc]", fontsize=labelsize)

    fig.savefig(figure_file, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run the demo and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "examples" / "demo_output"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cube_file = run_vof(args.output.resolve())
    figure_file = args.output / "demo.png"
    plot(cube_file, figure_file)
    print(f"Datacube: {cube_file}\nFigure:   {figure_file}")  # noqa: T201 - a script's final output


if __name__ == "__main__":
    main()
