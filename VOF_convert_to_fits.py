from astropy.io import fits
import numpy as np
import h5py
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


def convert_to_fits(spectra,output_filename,fov,observer_distance,obs_spatial_res_arcseconds,dnu_kmps):

    nx,ny,nspec = np.shape(spectra)

    #Get Observational Parameters
    fov_kpc = fov*u.kpc
    observer_distance = observer_distance * u.kpc#in kpc
    z = observer_distance * H0 / c

    dnu_mps = dnu_kmps * 1000.




    input_filename = "/Users/ctrapp/Documents/foggie_analysis/analysis_tools/tilted_ring_fits/NGC_2403_NA_CUBE_THINGS.fits"
    with fits.open(input_filename) as hdul:
        hdul.info()  # Show HDU list
        header = hdul[0].header
        data = hdul[0].data  # This is a NumPy array


# Step 4: Save to a new FITS file
#output_filename = "/Users/ctrapp/Documents/foggie_analysis/analysis_tools/tilted_ring_fits/"+gal_name+"_NHI18_unfiltered_mock_ifu.fits"

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
    header['OBSERVER'] = 'ctrapp  '

    hdu = fits.PrimaryHDU(data=new_image, header=header)
    hdu.writeto(output_filename, overwrite=True)

    print("SHAPE OF IMAGE =",nx,ny)
    print("bmaj=",bmaj)
    print("bmaj in arsec=",bmaj*3600)
    print("fov_kpc=",fov_kpc)
    print("arcsec per pixel=",fov_deg / nx * 3600)
    print("kpc per pixel=",fov_kpc / nx)

    print(f"\nSaved modified FITS file as {output_filename}")