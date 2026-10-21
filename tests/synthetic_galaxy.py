"""Write a small synthetic FIRE-like snapshot for the tests: a tilted, rotating disk galaxy.

The snapshot is split across two HDF5 files, like real multi-part FIRE snapshots, and is
stored in the snapshot's comoving, h-scaled units, so reading it back exercises the
readers' unit conversions. Everything returned in GalaxyTruth is in physical units
(kpc, km/s, 10^10 Msun), i.e. what VOF should recover.
"""

from dataclasses import dataclass, field

import h5py
import numpy as np

from vof.constants import proton_mass, unit_L

SNAPSHOT = 600
ASCALE = 0.9  # scale factor of the snapshot
HUBBLE = 0.7  # little h
N_METAL_FIELDS = 11  # FIRE stores total metallicity, He, then individual elements

N_PER_FILE = {
    0: [2200, 1800],
    1: [1000, 1000],
    4: [1100, 900],
}  # particle type -> count in each file
N_HALO_GAS = 400  # hot, diffuse gas in a 40 kpc sphere, so every sightline through the image hits some gas

CENTER = np.array([450.0, 470.0, 430.0])  # galaxy center, kpc
VEL_CENTER = np.array([50.0, -30.0, 20.0])  # bulk velocity of the galaxy, km/s
V_FLAT = 200.0  # asymptotic rotation speed, km/s
R_TURN = 2.5  # turnover radius of the rotation curve, kpc


def circular_velocity(R):
    """Return the disk's rotation speed (km/s) at cylindrical radius R (kpc)."""
    return V_FLAT * (1 - np.exp(-R / R_TURN))


def disk_axes(tilt_deg=30.0):
    """Return (Lhat, r0, y0): the disk spin axis and two in-plane unit vectors."""
    t = np.radians(tilt_deg)
    Lhat = np.array([0.0, -np.sin(t), np.cos(t)])  # z axis tilted toward -y
    r0 = np.array([1.0, 0.0, 0.0])
    return Lhat, r0, np.cross(Lhat, r0)


@dataclass
class GalaxyTruth:
    """The galaxy's true properties, in physical units, and where its files are."""

    snapdir: str
    stats_file: str
    Lhat: np.ndarray
    r0: np.ndarray
    center: np.ndarray = field(default_factory=lambda: CENTER.copy())
    vel_center: np.ndarray = field(default_factory=lambda: VEL_CENTER.copy())
    # particle type -> {"p", "v", "m", ...}, in file order
    particles: dict = field(default_factory=dict)


def _mirrored(pos, vel):
    """Add a (-p, -v) twin for every particle, so the center of mass and net momentum are exactly 0.

    Rotation is unchanged by the mirroring, and it keeps tests of the centering from
    depending on sampling noise.
    """
    return np.concatenate([pos, -pos]), np.concatenate([vel, -vel])


def _disk(rng, n, scale_length, scale_height, sigma_v, Lhat, r0, y0):
    """Return positions, velocities (galaxy frame), and radii of n particles in a rotating disk."""
    n //= 2  # the other half are mirrored twins
    R = np.minimum(rng.gamma(2.0, scale_length, n), 25.0)  # exponential surface density
    phi = rng.uniform(0, 2 * np.pi, n)
    z = rng.normal(0, scale_height, n)
    R_hat = np.cos(phi)[:, None] * r0 + np.sin(phi)[:, None] * y0
    phi_hat = np.cross(Lhat, R_hat)
    pos = R[:, None] * R_hat + z[:, None] * Lhat
    vel = circular_velocity(R)[:, None] * phi_hat + rng.normal(0, sigma_v, (n, 3))
    return *_mirrored(pos, vel), np.concatenate([R, R])


def _gas(rng, Lhat, r0, y0):
    """Return the gas fields (physical units): a cold, dense disk plus a hot halo."""
    n_disk = sum(N_PER_FILE[0]) - N_HALO_GAS
    pos, vel, R = _disk(rng, n_disk, 3.0, 0.2, 8.0, Lhat, r0, y0)
    n_H = 20.0 * np.exp(-R / 3.0)  # hydrogen number density, cm^-3
    hsml = np.minimum(0.1 * np.exp(R / 8.0), 0.8)  # smoothing length, kpc
    # internal energy, (km/s)^2 -> T of roughly 4000-7000 K
    u = rng.uniform(40, 70, n_disk)
    ne = rng.uniform(0.01, 0.1, n_disk)
    nh = rng.uniform(0.9, 1.0, n_disk)  # neutral hydrogen fraction

    # Hot, diffuse halo gas, uniform within 40 kpc (extends past the region VOF images)
    direction = rng.normal(size=(N_HALO_GAS, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    pos_h = 40.0 * rng.uniform(0, 1, N_HALO_GAS)[:, None] ** (1 / 3) * direction
    vel_h = rng.normal(0, 100, (N_HALO_GAS, 3))

    Z = np.zeros((sum(N_PER_FILE[0]), N_METAL_FIELDS))
    Z[:, 0] = rng.uniform(0.01, 0.03, len(Z))  # total metallicity
    Z[:, 1] = rng.uniform(0.25, 0.29, len(Z))  # helium
    Z[:, 2:] = rng.uniform(0, 1e-3, (len(Z), N_METAL_FIELDS - 2))
    X_H = 1 - Z[:, 0] - Z[:, 1]

    n_H = np.concatenate([n_H, np.full(N_HALO_GAS, 1e-4)])
    rho = n_H * proton_mass / X_H * unit_L**3  # cm^-3 -> 10^10 Msun / kpc^3
    gas = {
        "p": np.concatenate([pos, pos_h]),
        "v": np.concatenate([vel, vel_h]),
        "m": np.full(len(Z), 2.5e-4),
        "rho": rho,
        "h": np.concatenate(
            [hsml, np.full(N_HALO_GAS, 10.0)]
        ),  # diffuse gas has large smoothing lengths
        "u": np.concatenate([u, rng.uniform(1e4, 1e5, N_HALO_GAS)]),
        "ne": np.concatenate([ne, np.full(N_HALO_GAS, 1.2)]),
        "nh": np.concatenate([nh, np.full(N_HALO_GAS, 1e-4)]),
        "z": Z,
    }
    order = rng.permutation(len(Z))  # mix halo and disk particles across the two files
    return {k: v[order] for k, v in gas.items()}


def _dark_matter(rng):
    """Return positions/velocities (galaxy frame) of a roughly Hernquist-like halo."""
    n = sum(N_PER_FILE[1]) // 2  # the other half are mirrored twins
    s = np.sqrt(rng.uniform(0, 0.95, n))
    r = 15.0 * s / (1 - s)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    pos, vel = _mirrored(r[:, None] * direction, rng.normal(0, 120, (n, 3)))
    return {"p": pos, "v": vel, "m": np.full(2 * n, 5e-3)}


def write_synthetic_snapshot(snapdir, seed=7, tilt_deg=30.0, single_file=False):
    """Write snapshot_600.{0,1}.hdf5 (or snapshot_600.hdf5 if single_file) into snapdir; return the GalaxyTruth."""
    rng = np.random.default_rng(seed)
    Lhat, r0, y0 = disk_axes(tilt_deg)

    pos_s, vel_s, _R = _disk(rng, sum(N_PER_FILE[4]), 2.0, 0.3, 15.0, Lhat, r0, y0)
    particles = {
        0: _gas(rng, Lhat, r0, y0),
        1: _dark_matter(rng),
        4: {
            "p": pos_s,
            "v": vel_s,
            "m": np.full(len(pos_s), 5e-4),
            "age": rng.uniform(0.1, 0.9, len(pos_s)),
        },
    }
    for part in particles.values():  # galaxy frame -> simulation frame
        part["p"] = part["p"] + CENTER
        part["v"] = part["v"] + VEL_CENTER

    snapdir.mkdir(parents=True, exist_ok=True)
    totals = np.zeros(6, np.int64)
    for ptype, counts in N_PER_FILE.items():
        totals[ptype] = sum(counts)

    per_file = (
        {ptype: [sum(counts)] for ptype, counts in N_PER_FILE.items()}
        if single_file
        else N_PER_FILE
    )
    n_files = 1 if single_file else 2
    starts = dict.fromkeys(per_file, 0)
    for ifile in range(n_files):
        suffix = "" if single_file else f".{ifile}"
        with h5py.File(snapdir / f"snapshot_{SNAPSHOT}{suffix}.hdf5", "w") as f:
            this_file = np.zeros(6, np.int64)
            for ptype, counts in per_file.items():
                this_file[ptype] = counts[ifile]
            header = {
                "NumPart_ThisFile": this_file,
                "NumPart_Total": totals,
                "MassTable": np.zeros(6),
                "Time": ASCALE,
                "Redshift": 1 / ASCALE - 1,
                "BoxSize": 2000.0,
                "Omega0": 0.27,
                "OmegaLambda": 0.73,
                "HubbleParam": HUBBLE,
                "NumFilesPerSnapshot": n_files,
                "Flag_Sfr": 1,
                "Flag_Feedback": 1,
                "Flag_Cooling": 1,
                "Flag_StellarAge": 1,
                "Flag_Metals": N_METAL_FIELDS,
            }
            f.create_group("Header").attrs.update(header)

            for ptype, counts in per_file.items():
                sl = slice(starts[ptype], starts[ptype] + counts[ifile])
                starts[ptype] += counts[ifile]
                _write_particles(
                    f.create_group(f"PartType{ptype}"), ptype, particles[ptype], sl
                )

    stats_file = snapdir / f"stats_{SNAPSHOT:04d}.hdf5"
    return GalaxyTruth(str(snapdir), str(stats_file), Lhat, r0, particles=particles)


def _write_particles(group, ptype, part, sl):
    """Write one file's slice of a particle type, converting to snapshot units."""
    n = sl.stop - sl.start
    group["Coordinates"] = part["p"][sl] * HUBBLE / ASCALE  # comoving kpc/h
    group["Velocities"] = part["v"][sl] / np.sqrt(ASCALE)  # gadget velocity units
    group["Masses"] = part["m"][sl] * HUBBLE  # 10^10 Msun/h
    group["ParticleIDs"] = np.arange(sl.start, sl.stop) + 10**6 * ptype
    group["ParticleChildIDsNumber"] = np.zeros(n, np.int64)
    group["ParticleIDGenerationNumber"] = np.zeros(n, np.int64)
    if ptype == 0:
        group["Density"] = part["rho"][sl] * ASCALE**3 / HUBBLE**2  # comoving h^2 units
        group["SmoothingLength"] = part["h"][sl] * HUBBLE / ASCALE
        group["InternalEnergy"] = part["u"][sl]
        group["ElectronAbundance"] = part["ne"][sl]
        group["NeutralHydrogenAbundance"] = part["nh"][sl]
        group["Metallicity"] = part["z"][sl]
    if ptype == 4:
        group["StellarFormationTime"] = part["age"][sl]


def write_stats_file(truth, max_r=15.0, max_z=5.0):
    """Write a stats file holding the galaxy's true center, velocity, and orientation."""
    with h5py.File(truth.stats_file, "w") as f:
        f["pos_center"] = truth.center
        f["vel_center"] = truth.vel_center
        f["Lhat"] = truth.Lhat
        f["r0"] = truth.r0
        for key, value in {
            "max_x": max_r,
            "max_y": max_r,
            "max_z": max_z,
            "ascale": ASCALE,
        }.items():
            f[key] = value
