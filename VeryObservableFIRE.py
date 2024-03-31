import numpy as np
import h5py as h5py
import os.path
import time

from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool

from functools import partial

from VOF_GenerateSightlineFiles import GenerateSightlineFiles

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
            

def GenerateSyntheticImage(fileDir,statsDir, Nsnap, output,sightlineDir,
                            observerDistance, observerVelocity,
                            maxRadius,
                            noiseAmplitude,beamSize,targetBeamSize,Nsightlines1d,
                            phiObs,inclination,
                            speciesToRun,Nspec,bandwidth,
                            savePNG,createSightlineFiles=True,bandwidth_km_s=None,
    ):
                                        
    t1 = time.time()
    Nsnapstring = str(Nsnap)     
    snapDir = fileDir+Nsnapstring #Directory containing the actual snapshots
    statsDir = statsDir+Nsnapstring.zfill(4)+".hdf5" #Centering/orientation information.
    
    Nsightlines=Nsightlines1d*Nsightlines1d

    print("Generating Sightline Files...")
    observer_position = np.array([-observerDistance, phiObs, 0]) #in spherical coordinates
   # maxPhi = pi* maxRadius / observerDistance #Convert physical size of observation to radians on the sky
    maxPhi = 2 * maxRadius / observerDistance #Convert physical size of observation to radians on the sky

    maxTheta = maxPhi
    if maxTheta>pi: #You probably shouldn't ever look at an image this big anyway...
        maxTheta = pi
        
    maxima=[maxRadius,maxPhi,maxTheta]
    print("Beam size set to: ",beamSize /arcsec," ''")
    
    
    gaussianMat = Generate_PSF_Matrix(Nsightlines1d,beamSize,targetBeamSize,Nspec) #Precalculate PSF matrix

    #Predefine which particles belong to which sightline files to speed up parallelization. Can be re-used for observations from the same distance/inclination
    image=GenerateSightlineFiles(snapDir,Nsnapstring,statsDir,observer_position,observerVelocity,sightlineDir,maxima,beamSize,Nsightlines,phiObs = phiObs, inclination = inclination,speciesToRun=speciesToRun,Nspec=Nspec,bandwidth=bandwidth,targetBeamSize=targetBeamSize,noiseAmplitude=noiseAmplitude,gaussianMat=gaussianMat) 
        

        
    #### Start Paralellization ####
    runParallel=True  
    if True:                 
        hf = h5py.File(output+'_fullSpectra.hdf5','w')
        hf.create_dataset('spectra',data=image)
        hf.close()
        if savePNG: #Option to create a column density map to visualize results immediately
            fig = plt.figure()
            vmax = np.max(np.sum(image,2))
            vmin = vmax * np.power(10.0,-4.0)
            plt.imshow(np.sum(image,2),norm=LogNorm(vmin=vmin,vmax=vmax),cmap='inferno')
            plt.colorbar()
            plt.savefig(output+'_fullSpectra.png')
            plt.close()
            
            plt.figure()
            plt.imshow(-(np.argmax(image,axis=2)*400.0/77.0 - 200),vmin=-200,vmax=200,cmap='seismic')
            plt.colorbar()
            plt.savefig(output+'_DopplerVelEstimate.png')
            plt.close()

            
    print("Snapshot ",Nsnapstring," ran in ",time.time()-t1) 

   
   
def Generate_PSF_Matrix(Nsightlines1d,beamSize,targetBeamSize,Nspec):
    #Creates a gaussian matrix representing the psf that can be applied to every spectral entry
    #
    #For each bin convolve with psf(2-d gaussian kernel) and add result to modelImage 
    #lambda_k = stdv in k direction 
    #lambda_n = .stdv in n direction

    lambda_k = targetBeamSize / beamSize
    lambda_n = targetBeamSize / beamSize #way this is set up, should be about 1 px per beamsize...
    
    prefactor = 1/2/pi/lambda_k/lambda_n 
    
    gaussianMat = np.zeros((Nsightlines1d*2 , Nsightlines1d*2))
    x = np.indices(np.shape(gaussianMat))[0]
    y = np.indices(np.shape(gaussianMat))[1]
    
    xc = np.shape(gaussianMat)[0]/2
    yc = np.shape(gaussianMat)[1]/2
    
    gaussianMat = prefactor * np.exp(-0.5 * (np.power(x-xc,2)/lambda_k/lambda_k + np.power(y-yc,2)/lambda_n/lambda_n)) #add psf from pixel
    normCorrection = np.sum(gaussianMat)
    gaussianMat*=1/normCorrection #account for resolution limitations for a very small psf in this pixel space

    return np.repeat(gaussianMat[:,:,np.newaxis],Nspec,axis=2) #repeat for each spectral channel