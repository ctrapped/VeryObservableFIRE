#!/bin/bash

#load from h5py
#save  as .fits

parfile=$1
base_path=$2
filename=$3

if [ $parfile = "-1" ]; then
    parfile="par_things.par"
fi

if [ $base_path = "-1" ]; then
    base_path="/Users/ctrapp/Documents/GitHub/VeryObservableFIRE/tmp/"
fi

if [ $filename = "-1" ]; then
    filename="temp.fits"
fi

cd /Users/ctrapp/Documents/foggie_analysis/SoFiA-2-master

echo "Running Sofia on $filename"
./sofia "${parfile}" input.data="${base_path}${filename}"
