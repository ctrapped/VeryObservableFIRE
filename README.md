# VeryObservableFIRE

**V**ery**O**bservable**F**IRE (VOF) generates synthetic spectral datacubes from FIRE simulation snapshots, mimicking what a real telescope would observe. Given a snapshot, an observer distance/inclination/position angle, and a target line, VOF orients and centers the galaxy, traces sightlines through the gas, computes the resulting emission/absorption spectrum per sightline, and convolves the result with an instrument beam and noise model to produce a mock IFU-style datacube. This can also be run as a faster, optically thin projection to save time or on individual/limited sightlines.

Currently supported lines: **HI 21cm**, **H-alpha**, and the **Sodium I doublet** (Na I D1/D2).

If `runBinfire` is enabled, VOF also uses the bundled `Binfire` submodule to bin the true simulation gas onto the same projected grid (mass, radial/azimuthal mass flux, rotation curve), producing annotation files that pair with the synthetic images for neural network training (e.g. with CoNNGaFit). Alternatively, if `projectGasProperties` and `runAsOpticallyThinProjection` are enabled, VOF can project these properties in the same imaging pass.

## Requirements

Python 3 with:
- `numpy`, `scipy`, `h5py`, `matplotlib`, `joblib`
- `astropy`, `unyt` (only needed for the optional FITS conversion utility)


## How it works

1. **`VeryObservableFIRE.py`** — entry point. Loads a plain-text parameter file (via `LoadParamFile.LoadParams`), computes derived observation quantities (beam size, line frequency, bandwidth), and loops over the requested snapshot range calling `FireToDataset`.
2. **`VOF_ConvertDataset.py`** (`FireToDataset`) — for each snapshot, loops over every requested inclination × position angle pair. Optionally runs `Binfire` to produce annotation maps, then calls `VOF_GenerateSyntheticImage.py` to build the mock datacube.
3. **`VOF_GenerateSyntheticImage.py`** — sets up the observer geometry and calls `VOF_GenerateSightlines.py`.
4. **`VOF_GenerateSightlines.py`** — orients/centers the galaxy (`VOF_OrientGalaxy.py`), loads gas particles, casts a grid of sightlines, and (in parallel via `joblib`) computes each sightline's spectrum (`VOF_GetParticlesInSightline.py` → `VOF_GenerateSpectra.py`, using line parameters from `VOF_EmissionSpecies.py`). The resulting cube is smoothed to the target beam size and has noise added.
5. Output is written as HDF5 (`*_fullSpectra.hdf5`, containing `spectra`/`ideal_image`/`smooth_image` datasets, plus `fov_kpc`/`observer_distance_kpc`/`beam_arcsec`/`dnu_kmps` attrs recording the observation setup used to produce it) plus optional zeroth/first-moment PNG previews.

## Usage

### Primary way: parameter files

The main way to run VOF is by writing parameter file and passing its path on the command line. Start from `param_template.param` and copy it to a new file, e.g. `param_myrun.param`. It's a list of `key = value` lines (grouped under `# File Parameters` / `# Observer Parameters` / `# Run Parameters` comments, which are purely cosmetic) — see the file itself for the full list and what each key does.

A value can reference an **earlier** key in the same file:
- as a Python expression, e.g. `noiseAmplitude = 4e19/3.` (`pi`/`arcsec` are always available in expressions too)
- as a `{key}` placeholder inside a path string, e.g. `fileDir = /path/to/sims/{galName}/snapdir_`

Values are otherwise parsed as Python literals (numbers, `True`/`False`, `None`, `[1, 2, 3]` lists) where possible, and as plain strings otherwise (quotes are optional for simple strings, e.g. both `galName = m12m` and `galName = 'm12m'` work).

Edit the values, then run:

```bash
python VeryObservableFIRE.py param_myrun.param
```

(the `.param` extension is auto-appended if you omit it).

### Command-line overrides

`VeryObservableFIRE.py` also accepts up to four optional positional arguments that override values a parameter file would otherwise supply, useful for scripting batch runs without editing the file each time:

```bash
python VeryObservableFIRE.py <param_file> [galName] [minSnap] [maxSnap] [inclination]
```

- `galName` — overrides `galName`, including in any `{galName}` path templating elsewhere in the file.
- `minSnap` — overrides the starting snapshot number.
- `maxSnap` — overrides the ending snapshot number (defaults to `minSnap` if omitted, i.e. a single snapshot).
- `inclination` — if given, restricts the run to this single inclination instead of the `inclinations` list in the file.

Any argument left off falls back to the value defined in the parameter file. These are implemented in `LoadParamFile.LoadParams(path, galName=..., minSnap=..., maxSnap=..., inclination=...)`, which any other script can also call directly to read a `.param` file into a plain `dict`.

### Optional post-processing

`VOF_convert_to_fits.py` converts a VOF spectra datacube to a FITS cube, using an existing FITS file as a header template. It can be used as a library function (`convert_to_fits(...)`) or run directly from the command line:

```bash
python VOF_convert_to_fits.py <spectra.hdf5> <template.fits> <output.fits> \
    --fov <kpc> --observer-distance <kpc> --beam-arcsec <arcsec> --dnu-kms <km/s> [--observer-name <name>]
```

`<spectra.hdf5>` is a VOF output file (e.g. `*_fullSpectra.hdf5`, read from its `spectra` dataset); `<template.fits>` supplies the header keywords not otherwise set from the arguments above.

`RunSofiaForVOF.sh` runs the SoFiA-2 source finder on a converted cube:

```bash
./RunSofiaForVOF.sh <sofia_dir> <base_path> [parfile] [filename]
```

`sofia_dir` is your local SoFiA-2 install directory (containing the `sofia` executable), `base_path` is the directory holding the FITS cube, and `parfile`/`filename` default to `par_things.par`/`temp.fits` if omitted.

`VOF_ImageRotater.py` batch-augments a set of runs' outputs (rotations/flips of each datacube and its Binfire annotations), optionally denoising each cube via SoFiA-2 first. It reads `fov_kpc`/`observer_distance_kpc`/`beam_arcsec`/`dnu_kmps` back off each `*_fullSpectra.hdf5` (as written by `VOF_GenerateSyntheticImage.py`) rather than hardcoding them, and takes the batch selection and SoFiA-2 paths as CLI arguments (each defaulting to the values previously hardcoded in the script). Run it standalone like this to augment existing outputs after the fact:

```bash
python VOF_ImageRotater.py --data-root <dir> --gal-names m12m m12i --inclinations 50 60 --position-angles 0 45 90 135 180 225 270 315 --snapshots 600 \
    [--tags ""] [--masked-tags ""] [--denoise --sofia-dir <sofia_dir> --sofia-base-path <base_path> --template-fits <template.fits>]
```

`--data-root` is the base directory containing per-galaxy outputs matching the `output` directory layout written by the main pipeline. `--denoise` is a flag (off by default) that runs SoFiA-2 masking on each cube before rotating it; `--sofia-dir`/`--sofia-base-path`/`--template-fits` only matter when `--denoise` is set.

Each rotated/flipped variant it writes is also appended as a new row to the same `.csv` annotation file the main pipeline writes (via the same `AppendToAnnotationsFile` used by `VOF_ConvertDataset.py`).

## Parameters

Every key a `.param` file can set, grouped the same way as in `param_template.param`:

### File Parameters

| Parameter | Description |
|---|---|
| `galName` | Simulation name, as a string |
| `minSnap` | Starting snapshot number |
| `maxSnap` | Ending snapshot number |
| `fileDir` | Path to the directory with snapshots (should end without the trailing snapshot number) |
| `statsDir` | Path to a directory to store centering/orientation stats; created automatically if it doesn't exist |
| `output` | Directory to write outputs to |

### Observer Parameters

| Parameter | Description |
|---|---|
| `observerDistance` | Distance to the observer, in kpc |
| `observerVelocity` | Observer velocity, `[vx, vy, vz]` |
| `maxRadius` | Max radius from the disk center to image, in kpc |
| `maxHeight` | Max height above the disk plane to include, in kpc |
| `targetBeamSize` | Beam size of the instrument being modeled, in arcseconds |
| `Nsightlines1d` | Number of sightlines/pixels along one axis of the image |
| `phiObs` | Offset the image by this angle, in radians |
| `inclinations` | List of inclinations to image, in degrees |
| `position_angles` | List of position angles to image, in degrees |
| `speciesToRun` | Which line to model (`HI_21cm`, `h_alpha`, `NaI_D1`, or `NaI_D2`) |
| `res_km_s` | Spectral resolution, in km/s |
| `Nchannels` | Number of spectral channels (bandwidth = `Nchannels * res_km_s`) |
| `noiseAmplitude` | Noise amplitude added to the image |

### Run Parameters

| Parameter | Description |
|---|---|
| `runBinfire` | `True` to generate Binfire annotation files (mass flux, mass, rotation curve) |
| `runVOF` | `True` to generate the synthetic spectral datacubes |
| `savePng` | `True` to also save PNG previews of the images/annotations |
| `createMaskFromExistingStatsDir` | `True` to mask the previously-run galaxy, to find satellites/other galaxies in the snapshot |
| `num_cores` | Number of cores to use for parallel sightline generation; `None` to use all available |
| `projectGasProperties` | `True` to project the true gas mass/radial/rotational velocity directly in the same imaging pass (used together with `runAsOpticallyThinProjection`) |
| `runAsOpticallyThinProjection` | `True` for the fast optically-thin column-density projection (`DepositParticles`); `False` for full per-sightline spectral synthesis |
