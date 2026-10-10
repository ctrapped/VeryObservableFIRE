import numpy as np
import h5py as h5py
import os.path
import time

from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool

from functools import partial

from ProjectImage import ProjectImage

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

pi = np.pi
arcsec = (1. /60. / 60.) * pi/180.

image=0

Joules2eV = 6.241509*np.power(10.0,18.0)
meters2Kpc = 3.24078*np.power(10.0,-20.0)
Jy2SimUnits = np.power(10.0,-26.0) * Joules2eV / meters2Kpc / meters2Kpc
#J*s^-1*m^-2*Hz-1
SimUnits2Jy = 1.0 / Jy2SimUnits 


####Each thread generates a spectra for the assigned pixel, then convolves it with a Gaussian PSF at it's location within the image.
####Returns a matrix the size of the image, containing soley the PSF contribution from the assigned sightline.
####This allows the total image to be created from summing each thread contribution. 
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 03-10-2023
            

def GenerateSyntheticImage(config, fileDir, statsDir, Nsnap, output, inclination, position_angle, particleData=None):

    #Config values this function's own body needs. Everything else ProjectImage needs
    #is read directly out of config there instead of being passed through here.
    observerDistance = config['observerDistance']
    maxRadius = config['maxRadius']
    beamSize = config['beamSize']
    targetBeamSize = config['targetBeamSize']
    Nsightlines1d = config['Nsightlines1d']
    phiObs = config['phiObs']
    Nspec = config['Nspec']
    bandwidth_km_s = config['bandwidth_km_s']
    savePNG = config['savePng']
    projectGasProperties = config['projectGasProperties']

    t1 = time.time()
    Nsnapstring = str(Nsnap)
    snapDir = fileDir+Nsnapstring #Directory containing the actual snapshots
    statsDir = statsDir+Nsnapstring.zfill(4)+".hdf5" #Centering/orientation information.

    Nsightlines=Nsightlines1d*Nsightlines1d

    print("Generating Sightline Files...")
    observer_position = np.array([-observerDistance, phiObs, 0]) #in spherical coordinates
    maxPhi = 2 * maxRadius / observerDistance #Convert physical size of observation to radians on the sky

    maxTheta = maxPhi
    if maxTheta>pi: #You probably shouldn't ever look at an image this big anyway...
        maxTheta = pi

    maxima=[maxRadius,maxPhi,maxTheta]
    print("Pixel size set to: ",beamSize /arcsec," ''")

    #Predefine which particles belong to which sightline files to speed up parallelization. Can be re-used for observations from the same distance/inclination
    if projectGasProperties:
        ideal_image, smooth_image, noisy_image, mass_map, rMom_map, sMom_map, rotMom_map = ProjectImage(config, snapDir, Nsnapstring, statsDir, observer_position, maxima, Nsightlines, inclination, position_angle, particleData=particleData)
    else:
        ideal_image, smooth_image, noisy_image = ProjectImage(config, snapDir, Nsnapstring, statsDir, observer_position, maxima, Nsightlines, inclination, position_angle, particleData=particleData)



    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    hf = h5py.File(output+'_fullSpectra.hdf5','w')
    hf.create_dataset('spectra',data=noisy_image)
    hf.create_dataset('ideal_image',data=ideal_image)
    hf.create_dataset('smooth_image',data=smooth_image)
    hf.attrs['fov_kpc'] = 2*maxRadius
    hf.attrs['observer_distance_kpc'] = observerDistance
    hf.attrs['beam_arcsec'] = targetBeamSize / arcsec
    if bandwidth_km_s is not None:
        hf.attrs['dnu_kmps'] = bandwidth_km_s / Nspec
    if projectGasProperties:
        mass_map[mass_map==0]=1e-20
        hf.create_dataset('mass_annotation',data=mass_map)
        hf.create_dataset('radial_velocity_annotation',data=np.divide(rMom_map,mass_map))
        hf.create_dataset('cylindrial_radial_velocity_annotation',data=np.divide(sMom_map,mass_map))
        hf.create_dataset('rotational_velocity_annotation',data=np.divide(rotMom_map,mass_map))
        hf.attrs['annotation_mass_units'] = "Msun"
        hf.attrs['annotation_velocity_units'] = "km/s"
    hf.close()

    if savePNG: #Option to create a column density map to visualize results immediately
        plt.figure()
        m0=np.sum(noisy_image,2)
        vmax = np.max(m0)
        vmin = vmax * 1e-8
        plt.imshow(np.abs(m0),norm=LogNorm(vmin=vmin,vmax=vmax),cmap='inferno')
        plt.colorbar(label='Column Density [cm$^{-2}$]')
        plt.savefig(output+'_ZerothMomentMap.png')
        plt.close()

        plt.figure()
        spec=np.linspace(-bandwidth_km_s/2,bandwidth_km_s/2,Nspec)
        m1 = np.divide( np.sum( np.multiply(noisy_image,spec[None,None,:]), axis=2) , m0)
        plt.imshow(m1,vmin=-bandwidth_km_s/2,vmax=bandwidth_km_s/2,cmap='seismic')
        plt.colorbar(label='First Moment [km/s]')
        plt.savefig(output+'_FirstMomentMap.png')
        plt.close()

        if projectGasProperties:
            plt.figure()
    
            plt.imshow(mass_map,norm=LogNorm(),cmap='inferno')
            plt.colorbar(label='Projected Mass (Msun)')
            plt.savefig(output+'_ProjectedMass.png')
            plt.close()

            plt.figure()
            vmax = 150
            vmin = -vmax
            plt.imshow(np.divide(rMom_map,mass_map),vmin=vmin,vmax=vmax,cmap='seismic')
            plt.colorbar(label='Radial Velocity (km/s)')
            plt.savefig(output+'_RadialVelocity.png')
            plt.close()


            plt.figure()
            vmax = np.max(np.abs(rMom_map)) / 2.
            vmin = -vmax
            plt.imshow(rMom_map,vmin=vmin,vmax=vmax,cmap='seismic')
            plt.colorbar(label='Radial Momentum (Msun km/s)')
            plt.savefig(output+'_RadialMomentum.png')
            plt.close()

            plt.figure()
            vmax = 150
            vmin = -vmax
            plt.imshow(np.divide(sMom_map,mass_map),vmin=vmin,vmax=vmax,cmap='seismic')
            plt.colorbar(label='Cylindrical Radial Velocity (km/s)')
            plt.savefig(output+'_CylRadialVelocity.png')
            plt.close()

            plt.figure()
            vmax = 400
            vmin = 0
            plt.imshow(np.divide(rotMom_map,mass_map),vmin=vmin,vmax=vmax,cmap='inferno')
            plt.colorbar(label='Rotational Velocity (km/s)')
            plt.savefig(output+'_RotationalVelocity.png')
            plt.close()


    print("Snapshot ",Nsnapstring," ran in ",time.time()-t1)

