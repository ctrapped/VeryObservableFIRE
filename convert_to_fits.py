from astropy.io import fits
import numpy as np
import h5py
import os
import unyt as u

c = 3e5 * u.km / u.s  # km/s
H0 = 70. * u.km / u.s / u.Mpc  # km/s/Mpc Hubble's constant
h = 4.135667696e-15 * u.eV * u.s #eV * s
Mpc_to_m = 3.08e22
Mpc_to_cm = Mpc_to_m * 100
kpc_to_cm = Mpc_to_cm / 1000
kb = 8.617333262e-5 * u.eV / u.K
m_e = 9.1094*np.power(10.,-28.) * u.g #grams
e = 4.8032*np.power(10.,-10.) * u.statC #cm^(3/2) * g^(1/2) * s^(-1)
amu = 1.6735575*np.power(10.,-24) * u.g

arcsec_to_rad = 1./60./60. * np.pi/180.

import argparse


def convert_to_fits(spectra,output_filename,fov,observer_distance,obs_spatial_res_arcseconds,dnu_kmps,template_filename,observer_name="unknown"):

    nx,ny,nspec = np.shape(spectra)

    #Get Observational Parameters
    fov_kpc = fov*u.kpc
    observer_distance = observer_distance * u.kpc#in kpc
    z = observer_distance * H0 / c

    dnu_mps = dnu_kmps * 1000.

    with fits.open(template_filename) as hdul:
        header = hdul[0].header.copy()

    bmaj = obs_spatial_res_arcseconds / 3600.#0.001666666666666666
    bmin = obs_spatial_res_arcseconds / 3600.#0.001666666666666666 

    image_array = spectra  


    new_image = np.zeros((1,nspec,ny,nx))
    for ks in range(0,nspec):
        new_image[0,ks,:,:] = image_array[:,:,ks]

    observer_distance = (z * c / H0).in_units("kpc")

    fov_deg = fov_kpc / observer_distance.in_units('kpc').v * 180./np.pi

    print("Fov_kpc=",fov_kpc)
    print("fov_deg=",fov_deg)

    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['NAXIS3'] = nspec
          
    header['CRPIX1'] = int(nx/2)
    header['CDELT1'] = -fov_deg / nx
    header['CUNIT1'] = 'DEGREE            '

    header['CRPIX2'] = int(ny/2)
    header['CDELT2'] = fov_deg / ny
    header['CUNIT2'] = 'DEGREE            '

    header['CRVAL3'] = 0
    header['CRPIX3'] = int(nspec/2)
    header['CDELT3'] = -dnu_mps
    header['CUNIT3'] = 'M/S               ' 

    header['BMAJ'] = bmaj
    header['BMIN'] = bmin
    header['OBJECT'] = "TEMP"
    header['OBSERVER'] = observer_name

    hdu = fits.PrimaryHDU(data=new_image, header=header)
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    hdu.writeto(output_filename, overwrite=True)

    print("SHAPE OF IMAGE =",nx,ny)
    print("bmaj=",bmaj)
    print("bmaj in arsec=",bmaj*3600)
    print("fov_kpc=",fov_kpc)
    print("arcsec per pixel=",fov_deg / nx * 3600)
    print("kpc per pixel=",fov_kpc / nx)

    print(f"\nSaved modified FITS file as {output_filename}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Convert a VeryObservableFIRE spectra datacube to a FITS cube.")
    parser.add_argument("spectra_h5", help="Path to an HDF5 file with a 'spectra' dataset (e.g. a *_fullSpectra.hdf5 output from VeryObservableFIRE).")
    parser.add_argument("template_fits", help="Path to an existing FITS cube whose header is used as a template.")
    parser.add_argument("output_fits", help="Path to write the resulting FITS cube.")
    parser.add_argument("--fov", type=float, required=True, help="Field of view in kpc.")
    parser.add_argument("--observer-distance", type=float, required=True, help="Observer distance in kpc.")
    parser.add_argument("--beam-arcsec", type=float, required=True, help="Beam size in arcseconds.")
    parser.add_argument("--dnu-kms", type=float, required=True, help="Spectral resolution in km/s.")
    parser.add_argument("--observer-name", default="unknown", help="Value to write to the FITS OBSERVER header keyword.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with h5py.File(args.spectra_h5, 'r') as hf:
        spectra = np.array(hf['spectra'])
    convert_to_fits(spectra, args.output_fits, args.fov, args.observer_distance,
                     args.beam_arcsec, args.dnu_kms, args.template_fits,
                     observer_name=args.observer_name)