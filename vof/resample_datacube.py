"""Resample existing VOF datacubes to a different resolution/noise level

Written by Cameron Trapp (ctrapped@gmail.com).
"""

import argparse
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import scipy

from vof.constants import (
    arcsec,
)
from vof.logging_setup import add_logging_arguments, configure_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="vof-resample",
        description="Resample existing VOF images to differet target resolutions/sensitivities.",
    )
    parser.add_argument(
        "image_dir", help="Path to the image file/directory"
    )
    parser.add_argument(
        "output_dir", help="Path to the output directory"
    )
    parser.add_argument(
        "--spatial_resampling_factor",
        type=int,
        default=1,
        help="By what factor to reduce the number of pixels.",
    )
    parser.add_argument(
        "--target_res",
        type=float,
        default=None,
        help="What new resolution in arcseconds?",
        required=True
    )

    parser.add_argument(
        "--target_sensitivity",
        type=float,
        default=None,
        help="What new sensitivity in cm^-2?",
        required=True
    )
    add_logging_arguments(parser)
    return parser.parse_args()



def downsample(img, factor,method='mean'):
    '''Reshape an input datacube or image into a coarser version'''
    ndim = np.size(img.shape)
    if ndim<2 or ndim>3:
        raise ValueError(
            f"Image must have 2 or 3 dimensions, not {ndim}."
        )
    if method not in ['mean','sum']:
        raise ValueError("Downsampling only defined for mean and sum.")
    nx, ny = img.shape[:2]
    if nx % factor or ny % factor:
        raise ValueError(
            f"Image size {nx}x{ny} is not divisible by the resampling factor {factor}"
        )

    if ndim==3:
        nspec = img.shape[2]
        # Reshape into blocks and average along the block axes
        reshaped = img.reshape(nx // factor, factor, ny // factor, factor, nspec)
    elif ndim==2:
        # Reshape into blocks and average along the block axes
        reshaped = img.reshape(nx // factor, factor, ny // factor, factor)

    if method=='mean': return reshaped.mean(axis=(1, 3)).astype(img.dtype)
    if method=='sum':  return reshaped.sum(axis=(1, 3)).astype(img.dtype)

def resample_datacube(image_dir,
                      output_dir,
                      spatial_resampling_factor,
                      new_noise_level,
                      new_resolution_arcsec,
                      force_overwrite=False):
    '''
    Function that takes an image and downsamples it
    spatially by spatial_resampling_factor. This uses the ideal images, so noise and smoothing is
    recomputed and set by new_noise_level and new_resolution_arcsec. This is used to create images
    for new survey parameters without rerunning the whole vof pipeline.
    '''
    
    if Path(output_dir).resolve() == Path(image_dir).resolve() and not force_overwrite:
        raise ValueError(
            "Cannot overwrite original image."
        )
    with h5py.File(image_dir,'r') as hfi:
        ideal_image = hfi['ideal_image'][...]
        fov_kpc = hfi.attrs['fov_kpc']
        observer_distance_kpc = hfi.attrs['observer_distance_kpc']
        pixel_size_arcsec = fov_kpc / observer_distance_kpc / arcsec / np.shape(ideal_image)[0]
        dnu_kmps = hfi.attrs.get('dnu_kmps')

        annotation_data_exists = "mass_annotation" in hfi

        if annotation_data_exists:
           mass_annotation = hfi['mass_annotation'][...]
           radial_velocity_annotation = hfi['radial_velocity_annotation'][...]
           cylindrical_radial_velocity_annotation = hfi['cylindrical_radial_velocity_annotation'][...]
           rotational_velocity_annotation = hfi['rotational_velocity_annotation'][...]
           mass_units = hfi.attrs.get('annotation_mass_units')
           velocity_units = hfi.attrs.get('annotation_velocity_units')
        else:
            logger.debug(
                "No annotation datasets in: %s",
                image_dir,
            )

    #Smooth the image with new beam
    sigma = (
        new_resolution_arcsec / pixel_size_arcsec / (2 * np.sqrt(2 * np.log(2)))
    )
    smoothed_image = scipy.ndimage.gaussian_filter(ideal_image, sigma=sigma, axes=[0, 1])

    #Downsample the image
    resampled_image = downsample(smoothed_image, spatial_resampling_factor,'mean')

    # Add noise here
    if new_noise_level > 0:
        noise_profile = np.random.normal(
            0, new_noise_level, np.shape(resampled_image)
        )  # Create noise profile scaled by the downsampling we are doing
        noise_profile = scipy.ndimage.gaussian_filter(noise_profile, sigma=sigma / spatial_resampling_factor, axes=[0, 1])
        noise_profile = (
            noise_profile * new_noise_level / np.std(noise_profile)
        )  # Renormalize noise
        noisy_image = np.add(resampled_image, noise_profile)
    else:  # Noiseless; renormalizing zero noise would divide by zero
        noisy_image = resampled_image

    if annotation_data_exists:
        resampled_mass = downsample(mass_annotation, spatial_resampling_factor, 'sum')

        resampled_radial_velocity = downsample(np.multiply(mass_annotation,radial_velocity_annotation), spatial_resampling_factor, 'sum')
        resampled_radial_velocity = np.divide(resampled_radial_velocity , resampled_mass, out=np.zeros_like(resampled_mass), where=resampled_mass > 0)

        resampled_rotational_velocity = downsample(np.multiply(mass_annotation,rotational_velocity_annotation), spatial_resampling_factor, 'sum')
        resampled_rotational_velocity = np.divide(resampled_rotational_velocity , resampled_mass, out=np.zeros_like(resampled_mass), where=resampled_mass > 0)

        resampled_cyl_radial_velocity = downsample(np.multiply(mass_annotation,cylindrical_radial_velocity_annotation), spatial_resampling_factor, 'sum')
        resampled_cyl_radial_velocity = np.divide(resampled_cyl_radial_velocity , resampled_mass, out=np.zeros_like(resampled_mass), where=resampled_mass > 0)

    Path(output_dir).parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(output_dir,'w') as hfo:
        hfo.create_dataset('spectra', data=noisy_image)
        hfo.attrs["fov_kpc"] = fov_kpc
        hfo.attrs["observer_distance_kpc"] = observer_distance_kpc
        hfo.attrs["beam_arcsec"] = new_resolution_arcsec
        if dnu_kmps is not None:
            hfo.attrs["dnu_kmps"] = dnu_kmps
        if annotation_data_exists:
            hfo.create_dataset("mass_annotation", data=resampled_mass)
            hfo.create_dataset("radial_velocity_annotation", data=resampled_radial_velocity)
            hfo.create_dataset("cylindrical_radial_velocity_annotation", data=resampled_cyl_radial_velocity)
            hfo.create_dataset("rotational_velocity_annotation", data=resampled_rotational_velocity)
            hfo.attrs["annotation_mass_units"] = mass_units
            hfo.attrs["annotation_velocity_units"] = velocity_units



def main():
    """Run the pipeline for every image in the directory."""
    startTime = time.time()
    args = parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)
    
    data_path = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    if data_path.is_dir():
        for path in sorted(data_path.glob("*.hdf5")):
            try:
                resample_datacube(path,output_dir / path.name,args.spatial_resampling_factor,args.target_sensitivity,args.target_res)
            except (OSError, KeyError, ValueError) as err:
                logger.warning(
                    "Could not process: %s: %s",
                    path,
                    err
                )
    else:
        resample_datacube(data_path,output_dir / data_path.name,args.spatial_resampling_factor,args.target_sensitivity,args.target_res)



    logger.info("Finished all images in %.1f s", time.time() - startTime)


if __name__ == "__main__":
    main()
