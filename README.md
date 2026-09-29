# VeryObservableFIRE

**V**ery**O**bservable**F**IRE (VOF) generates synthetic spectral datacubes from FIRE simulation snapshots, mimicking what a real telescope would observe. Given a snapshot, an observer distance/inclination/position angle, and a target line, VOF orients and centers the galaxy, traces sightlines through the gas, computes the resulting emission/absorption spectrum per sightline, and convolves the result with an instrument beam and noise model to produce a mock IFU-style datacube.

Currently supported lines: **HI 21cm**, **H-alpha**, and the **Sodium I doublet** (Na I D1/D2).

If `runBinfire` is enabled, VOF also uses the bundled `Binfire` submodule to bin the true simulation gas onto the same projected grid (mass, radial/azimuthal mass flux, rotation curve), producing annotation files that pair with the synthetic images for neural network training (e.g. with CoNNGaFit).

## Requirements

Python 3 with:
- `numpy`, `scipy`, `h5py`, `matplotlib`, `joblib`
- `astropy`, `unyt` (only needed for the optional FITS conversion utility)


## How it works

1. **`VeryObservableFIRE.py`** — entry point. Loads a parameter module, computes derived observation quantities (beam size, line frequency, bandwidth), and loops over the requested snapshot range calling `FireToDataset`.
2. **`VOF_ConvertDataset.py`** (`FireToDataset`) — for each snapshot, loops over every requested inclination × position angle pair. Optionally runs `Binfire` to produce annotation maps, then calls `VOF_GenerateSyntheticImage.py` to build the mock datacube.
3. **`VOF_GenerateSyntheticImage.py`** — sets up the observer geometry and calls `VOF_GenerateSightlines.py`.
4. **`VOF_GenerateSightlines.py`** — orients/centers the galaxy (`VOF_OrientGalaxy.py`), loads gas particles, casts a grid of sightlines, and (in parallel via `joblib`) computes each sightline's spectrum (`VOF_GetParticlesInSightline.py` → `VOF_GenerateSpectra.py`, using line parameters from `VOF_EmissionSpecies.py`). The resulting cube is smoothed to the target beam size and has noise added.
5. Output is written as HDF5 (`*_fullSpectra.hdf5`, containing `spectra`/`ideal_image`/`smooth_image` datasets, plus `fov_kpc`/`observer_distance_kpc`/`beam_arcsec`/`dnu_kmps` attrs recording the observation setup used to produce it) plus optional zeroth/first-moment PNG previews.
6. If `runDataAugmentation` is enabled, each image is immediately rotated/flipped via `VOF_ImageRotater.py` (`RotateData`, default angles `[0,90,180,270]`, no SoFiA-2 denoising) and the augmented variants are appended as new rows to the same annotation CSVs `FireToDataset` writes — this requires `runBinfire` and `runVOF` to also be enabled for that run, since it needs both the annotation files and the image just generated.

## Usage

### Primary way: parameter files

The main way to run VOF is by writing a parameter file and passing its module name on the command line. Start from `param_template.py` and copy it to a new file, e.g. `param_myrun.py`. Each parameter file defines three functions:

- **`LoadFileInfo()`** — simulation name, snapshot range, and paths to the snapshot directory, a stats-cache directory, and the output directory.
- **`LoadObserverInfo()`** — observer distance/velocity, field of view (`maxRadius`/`maxHeight`), target beam size, sightline grid resolution (`Nsightlines1d`), the list of `inclinations` and `position_angles` to render, which line to model (`speciesToRun`), spectral bandwidth/resolution, and noise amplitude.
- **`LoadParameters()`** — run-mode switches: whether to (re)build annotation files, whether to run Binfire and/or VOF, whether to save PNG previews, which annotation quantities to write, and whether to run data augmentation (`runDataAugmentation`).

Edit the values inside those functions, then run:

```bash
python VeryObservableFIRE.py param_myrun
```

(using the module name, not the filename — omit the `.py`).

### Command-line overrides

`VeryObservableFIRE.py` also accepts up to four optional positional arguments that override values a parameter file would otherwise supply, useful for scripting batch runs without editing the file each time:

```bash
python VeryObservableFIRE.py <param_module> [galName] [minSnap] [maxSnap] [inclination]
```

- `galName` — overrides the simulation name from `LoadFileInfo`.
- `minSnap` — overrides the starting snapshot number.
- `maxSnap` — overrides the ending snapshot number (defaults to `minSnap` if omitted, i.e. a single snapshot).
- `inclination` — if given, restricts the run to this single inclination instead of the list in `LoadObserverInfo`.

Any argument left off falls back to the value defined in the parameter file.

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

`VOF_ImageRotater.py` batch-augments a set of runs' outputs (rotations/flips of each datacube and its Binfire annotations), optionally denoising each cube via SoFiA-2 first. It reads `fov_kpc`/`observer_distance_kpc`/`beam_arcsec`/`dnu_kmps` back off each `*_fullSpectra.hdf5` (as written by `VOF_GenerateSyntheticImage.py`) rather than hardcoding them, and takes the batch selection and SoFiA-2 paths as CLI arguments (each defaulting to the values previously hardcoded in the script). This is the same augmentation `runDataAugmentation` runs automatically per-image during the main pipeline; run it standalone like this to re-augment existing outputs, use non-default rotation angles, or apply SoFiA-2 denoising (not exposed via `runDataAugmentation`):

```bash
python VOF_ImageRotater.py --data-root <dir> --gal-names m12m m12i --inclinations 50 60 --position-angles 0 45 90 135 180 225 270 315 --snapshots 600 \
    [--tags ""] [--masked-tags ""] [--denoise --sofia-dir <sofia_dir> --sofia-base-path <base_path> --template-fits <template.fits>]
```

`--data-root` is the base directory containing per-galaxy outputs; each combination is expected at `<data-root>/<gal-name>/vof_outputs/i<inclination>/training/`, matching the `output` directory layout written by the main pipeline. `--denoise` is a flag (off by default) that runs SoFiA-2 masking on each cube before rotating it; `--sofia-dir`/`--sofia-base-path`/`--template-fits` only matter when `--denoise` is set.

Each rotated/flipped variant it writes is also appended as a new row to the same `.csv` annotation file the main pipeline writes (via the same `AppendToAnnotationsFile` used by `VOF_ConvertDataset.py`).
