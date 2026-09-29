#!/bin/bash
# Run the SoFiA-2 source finder on a FITS cube produced by VOF_convert_to_fits.py

set -euo pipefail

usage() {
    echo "Usage: $0 <sofia_dir> <base_path> [parfile] [filename]" >&2
    echo "  sofia_dir  Path to the SoFiA-2 install directory (contains the 'sofia' executable)" >&2
    echo "  base_path  Directory containing the input FITS cube" >&2
    echo "  parfile    SoFiA parameter file, relative to sofia_dir (default: par_things.par)" >&2
    echo "  filename   FITS filename within base_path (default: temp.fits)" >&2
    exit 1
}

if [ "$#" -lt 2 ]; then
    usage
fi

sofia_dir=$1
base_path=$2
parfile=${3:-par_things.par}
filename=${4:-temp.fits}

cd "$sofia_dir"

echo "Running Sofia on $filename"
./sofia "$parfile" "input.data=${base_path}${filename}"
