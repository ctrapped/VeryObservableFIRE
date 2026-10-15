# VeryObservableFIRE

**V**ery**O**bservable**F**IRE (VOF) generates synthetic spectral datacubes from FIRE simulation snapshots, mimicking what a real telescope would observe. Given a snapshot, an observer distance/inclination/position angle, and a target line, VOF orients and centers the galaxy, traces sightlines through the gas, computes the resulting emission/absorption spectrum per sightline, and convolves the result with an instrument beam and noise model to produce a mock IFU-style datacube. This can also be run as a faster, optically thin projection to save time or on individual/limited sightlines.

Currently supported lines: **HI 21cm**, **H-alpha**, and the **Sodium I doublet** (Na I D1/D2).

If `runBinfire` is enabled, VOF also uses the bundled `Binfire` submodule to bin the true simulation gas onto the same projected grid (mass, radial/azimuthal mass flux, rotation curve), producing annotation files that pair with the synthetic images for neural network training (e.g. with CoNNGaFit). Alternatively, if `projectGasProperties` and `runAsOpticallyThinProjection` are enabled, VOF can project these properties in the same imaging pass.

## Installation

Requires Python 3.9+. From the repository root:

```bash
pip install -e .            # core pipeline
pip install -e ".[fits]"    # + FITS export and data augmentation (astropy, unyt)
pip install -e ".[dev]"     # + development tools (ruff, pytest)
```

This installs the `vof` package and three commands: `vof`, `vof-to-fits`, and `vof-augment`. The `-e` (editable) install links to this folder rather than copying it, so code changes and `git pull`s take effect without reinstalling.

### Running without installing

If you can't (or would rather not) `pip install`, install the dependencies listed in `pyproject.toml` yourself and run the package as a module from the repository root:

```bash
cd /path/to/VeryObservableFIRE
python -m vof param_myrun.param                    # instead of: vof
python -m vof.ConvertToFits ...                    # instead of: vof-to-fits
python -m vof.DataAugmentation ...                 # instead of: vof-augment
```

To run from a different directory, add the repository root to `PYTHONPATH` first (`export PYTHONPATH=/path/to/VeryObservableFIRE`). Running a file directly (e.g. `python vof/VeryObservableFIRE.py`) won't work, because the modules import each other as `vof.<module>`.

## Repository layout

```
configs/
  param_template.param        Starting point for a run's parameter file
  examples/                   Parameter files from real runs
vof/
  VeryObservableFIRE.py       `vof` entry point: loads the config, derives observation parameters, loops over snapshots
  LoadParamFile.py            Parser for the plain-text .param format
  MakeDataset.py              Per-snapshot driver: loads gas, runs Binfire, loops over inclination x position angle
  OrientGalaxy.py             Centering and rotating the galaxy into the observer's frame
  GenerateSyntheticImage.py   Builds one mock observation and writes HDF5/PNG outputs
  ProjectImage.py             Image formation: optically thin particle deposition (sparse stamping + FFT convolution)
                              or full per-sightline spectral synthesis (parallel via joblib), then beam smoothing + noise
  CalcSightlineSpectrum.py,
  GenerateSpectra.py          Emission/absorption spectrum along a sightline
  EmissionSpecies.py          Line parameters (HI 21cm, H-alpha, Na I D1/D2)
  CalcRotationCurve.py        Rotation curve, for observers placed inside the galaxy
  WriteAnnotations.py         Writes Binfire annotation files and previews
  ConvertToFits.py            `vof-to-fits`: HDF5 datacube -> FITS cube
  DataAugmentation.py         `vof-augment`: rotation/flip augmentation, optional SoFiA-2 denoising
  RunSofiaForVOF.sh           Wrapper used by augmentation to run SoFiA-2
  io/                         FIRE snapshot readers and loaders (LoadData.py, readsnap_VOF.py)
  Binfire/                    Binfire: centering (shrinking sphere), disk orientation, and binned
                              ground-truth maps (mass, radial/azimuthal mass flux, rotation)
  EnergyLevels/               NIST Na I energy levels
```

## How it works

1. **`vof.VeryObservableFIRE`** loads a parameter file (`vof.LoadParamFile.LoadParams`), computes derived observation quantities (beam size, line frequency, bandwidth), and calls `vof.MakeDataset.MakeDataset` for each snapshot in the requested range.
2. **`vof.MakeDataset.MakeDataset`** loads the gas particles. If `runBinfire`/`projectGasProperties` is set, it runs Binfire, which also centers the galaxy and writes the stats file the first time a snapshot is processed. It then loops over every inclination x position angle pair, rotates the galaxy into the observer's frame (`vof.OrientGalaxy`), and calls `vof.GenerateSyntheticImage.GenerateSyntheticImage`.
3. **`vof.ProjectImage.ProjectImage`** forms the image cube, either with the fast optically thin projection (`DepositParticles`) or with full per-sightline spectral synthesis (`vof.CalcSightlineSpectrum` -> `vof.GenerateSpectra`, using line parameters from `vof.EmissionSpecies`). The cube is then smoothed to the target beam size and has noise added.
4. The output is written as HDF5 (`*_fullSpectra.hdf5`). It holds the `spectra` and `ideal_image` datasets, plus the projected `*_annotation` maps when `projectGasProperties` is set. The `fov_kpc`/`observer_distance_kpc`/`beam_arcsec`/`dnu_kmps` attributes record the observation setup used. Zeroth/first-moment PNG previews are optional.

## Usage

### Parameter files

The main way to run VOF is to write a parameter file and pass its path on the command line. Copy `configs/param_template.param` to a new file, e.g. `param_myrun.param`. It's a list of `key = value` lines, grouped under `# File Parameters` / `# Observer Parameters` / `# Run Parameters` comments (the grouping is purely cosmetic). See the file itself for the full list and what each key does.

A value can reference an **earlier** key in the same file:
- as a Python expression, e.g. `noiseAmplitude = 4e19/3.` (`pi`/`arcsec` are always available in expressions too)
- as a `{key}` placeholder inside a path string, e.g. `fileDir = /path/to/sims/{galName}/snapdir_`

Otherwise, values are parsed as Python literals where possible (numbers, `True`/`False`, `None`, `[1, 2, 3]` lists) and as plain strings when not. Quotes are optional for simple strings: both `galName = m12m` and `galName = 'm12m'` work.

Edit the values, then run:

```bash
vof param_myrun.param          # or: python -m vof param_myrun.param
```

(the `.param` extension is added automatically if you leave it off).

### Command-line overrides

Optional flags override values from the parameter file, which is useful for scripting batch runs without editing the file each time:

```bash
vof <param_file> [--galName NAME] [--minSnap N] [--maxSnap N] [--inclination DEG]
```

- `--galName` overrides `galName`, including in any `{galName}` path templating elsewhere in the file.
- `--minSnap` overrides the starting snapshot number.
- `--maxSnap` overrides the ending snapshot number. If omitted it defaults to `--minSnap`, i.e. a single snapshot.
- `--inclination` restricts the run to this one inclination instead of the `inclinations` list in the file.

Any flag left off falls back to the value in the parameter file. These overrides are implemented in `vof.LoadParamFile.LoadParams(path, galName=..., minSnap=..., maxSnap=..., inclination=...)`, which other scripts can also call directly to read a `.param` file into a plain `dict`.

### Optional post-processing

These require the `fits` extra (`pip install -e ".[fits]"`).

`vof-to-fits` converts a VOF spectra datacube to a FITS cube, using an existing FITS file as a header template. It is also available as a library function (`vof.ConvertToFits.ConvertToFits(...)`):

```bash
vof-to-fits <spectra.hdf5> <template.fits> <output.fits> \
    --fov <kpc> --observer-distance <kpc> --beam-arcsec <arcsec> --dnu-kms <km/s> [--observer-name <name>]
```

`<spectra.hdf5>` is a VOF output file (e.g. `*_fullSpectra.hdf5`, read from its `spectra` dataset); `<template.fits>` supplies the header keywords not otherwise set from the arguments above.

`vof-augment` batch-augments a set of runs' outputs, writing rotations/flips of each datacube and its Binfire annotations. It can optionally denoise each cube with the SoFiA-2 source finder first, via `vof/RunSofiaForVOF.sh`. It reads `fov_kpc`/`observer_distance_kpc`/`beam_arcsec`/`dnu_kmps` back off each `*_fullSpectra.hdf5` rather than hardcoding them:

```bash
vof-augment --data-root <dir> --gal-names m12m m12i --inclinations 50 60 --position-angles 0 45 90 135 180 225 270 315 --snapshots 600 \
    [--tags ""] [--masked-tags ""] [--denoise --sofia-dir <sofia_dir> --sofia-base-path <base_path> --template-fits <template.fits>]
```

`--data-root` is the base directory containing per-galaxy outputs, laid out the way the main pipeline writes its `output` directory. `--denoise` is a flag (off by default) that runs SoFiA-2 masking on each cube before rotating it. `--sofia-dir` is your local SoFiA-2 install (containing the `sofia` executable); it and `--sofia-base-path`/`--template-fits` only matter when `--denoise` is set.

## Parameters

Every key a `.param` file can set, grouped the same way as in `configs/param_template.param`:

### File Parameters

| Parameter | Description |
|---|---|
| `galName` | Simulation name, as a string |
| `minSnap` | Starting snapshot number |
| `maxSnap` | Ending snapshot number |
| `snapStep` | Step between snapshots to run |
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
| `onlySaveFirstPng` | `True` to only save PNG previews for the first orientation of each snapshot |
| `createMaskFromExistingStatsDir` | `True` to mask the previously-run galaxy, to find satellites/other galaxies in the snapshot |
| `num_cores` | Number of cores to use for parallel sightline generation; `None` to use all available |
| `projectGasProperties` | `True` to project the true gas mass/radial/rotational velocity directly in the same imaging pass (used together with `runAsOpticallyThinProjection`) |
| `runAsOpticallyThinProjection` | `True` for the fast optically-thin column-density projection (`DepositParticles`); `False` for full per-sightline spectral synthesis |
