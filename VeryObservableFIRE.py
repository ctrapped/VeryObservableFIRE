import numpy as np
import h5py as h5py
import os.path
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import math
import sys
import time
import scipy.stats as stats

from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool

from functools import partial

from VeryObservableFIRE.VOF_preprocessing_GenerateSightlineFiles import GenerateSightlineFiles
from VeryObservableFIRE.VOF_preprocessing_GenerateSightlineFiles import GenerateHVCSightlineFiles

from VeryObservableFIRE.VOF_preprocessing_parallel_iterate_sightlines import IterateSightlines
from VeryObservableFIRE.VOF_preprocessing_parallel_iterate_sightlines import MakeMassImage

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

def foo(thread_id,snapDir, sightlineDir, Nsnapstring,statsDir,speciesToRun,beamSize,targetBeamSize,Nspec,bandwidths,output,Nsightlines1d):
    #print("Thread ",thread_id,": is running...")
    t0=time.time()
    outputDir = output + "_spectra"+str(thread_id)+".hdf5"
    threadImage=np.zeros((Nsightlines1d,Nsightlines1d,Nspec))
    ix=thread_id % Nsightlines1d
    iy = int(np.floor(thread_id / Nsightlines1d))
    #sightlineDir_thread = sightlineDir + "_sightline"+str(thread_id)+".hdf5"
    sightlineDir_thread = sightlineDir + "_allSightlines.hdf5"

    #CT: update for iterateSightlines to return this and not save an hdf5 file?
    threadImage[ix,iy,:]=IterateSightlines(thread_id,snapDir,Nsnapstring,statsDir,output,sightlineDir_thread,speciesToRun,beamSize,Nspec,bandwidths)
   # print("IN VOF, SUM OF SIGHTLINE SPECTRUM=",np.sum(threadImage[ix,iy,:]))
    threadImage[ix,iy,:] = threadImage[ix,iy,:] * SimUnits2Jy
   # print("Mean , Max of threadImage for ",thread_id,"= ",np.mean(threadImage),",",np.max(threadImage))
    noiseAmplitude = 0.0004 * beamSize / targetBeamSize#*Jy2SimUnits#Jy
    noiseProfile = np.random.normal(0, noiseAmplitude, np.shape(threadImage)[2])
    #print("Sum of noiseProfile=",np.sum(noiseProfile))
    
    threadImage = ConvolveWithPSF(threadImage,ix,iy,beamSize,targetBeamSize) 
    #print("Mean , Max of threadImage for ",thread_id," after Convolution= ",np.mean(threadImage),",",np.max(threadImage))
    threadImage[ix,iy,:] = np.add(threadImage[ix,iy,:] , noiseProfile) 
    #print("Mean , Max of threadImage for ",thread_id," after Noise= ",np.mean(threadImage),",",np.max(threadImage))

    #print("IN VOF, SUM OF SIGHTLINE SPECTRUM POST CONVOLVING=",np.sum(threadImage[ix,iy,:]))

    #print("NaNs after convolving:",np.isnan(threadImage))

    #hf = h5py.File(output+'.hdf5','r+')
    #data = hf['spectra']
    #data[...] = np.add( np.array(hf['spectra']) , threadImage )
    #hf.close()

    
    #global image
    #image = np.add(image,threadImage)
    print("Thread ",thread_id,": finished sightline in ",time.time()-t0)
    return threadImage
   # hf_thread = h5py.File(output+'thread_'+str(thread_id)+'.hdf5','w')
    #hf_thread.create_dataset('spectra',data=threadImage)
    #hf_thread.close()       
    #print("Thread ",thread_id,": finished sightline in ",time.time()-t0)


def GenerateSyntheticImage(fileDir,statsDir, Nsnap, outputs,sightlineDir,
                            observerDistance, observerVelocity,
                            maxRadius,
                            beamSize,targetBeamSize,Nsightlines1d,
                            phiObs,inclination,
                            speciesToRun,Nspec,bandwidths,
                            savePNG,createSightlineFiles=True,bandwidth_km_s=None,
    ):
                                        
    t1 = time.time()
    Nsnapstring = str(Nsnap)     
    snapDir = fileDir+Nsnapstring
    statsDir = statsDir+Nsnapstring.zfill(4)+".hdf5" #CT: update this...
    
    Nsightlines=Nsightlines1d*Nsightlines1d

    print("Generating Sightline Files...")
    observer_position = np.array([[-observerDistance, phiObs, 0]]) #in spherical coordinates
    maxPhi = pi* maxRadius / observerDistance
    maxTheta = maxPhi ##CT: is this factor reasonable?
    if maxTheta>pi:
        maxTheta = maxPhi/2
    print("Maxima are:",maxPhi,maxTheta)
    maxima=[maxRadius,maxPhi,maxTheta]
    #beamSize = maxPhi / Nsightlines1d
    print("Beam size set to: ",beamSize /arcsec," ''")
    #GenerateHVCSightlineFiles(snapDir,Nsnapstring,statsDir,observer_position,observer_velocity,outputs,maxima,beamSize,Nsightlines) 
    if createSightlineFiles:
        GenerateSightlineFiles(snapDir,Nsnapstring,statsDir,observer_position,observerVelocity,sightlineDir,maxima,beamSize,Nsightlines,phiObs = phiObs, inclination = inclination) 
    #### Start Paralellization ####
    runParallel=True
    
    #global image
    #image=np.zeros((Nsightlines1d, Nsightlines1d , Nspec))
    
    makeMassImage=True
    #if makeMassImage:
    #    MakeMassImage(snapDir,Nsnapstring,statsDir,Nsightlines1d,outputs+"totalMass.png", sightlineDir + "_sightline",speciesToRun)
        
    
    if (runParallel):
        print("Setting up Parallel Run...")
        num_cores = multiprocessing.cpu_count()-1
        #if (num_cores>4):
        #    num_cores=4
        output = outputs

        foo_ = partial(foo, snapDir=snapDir, sightlineDir=sightlineDir, Nsnapstring=Nsnapstring, statsDir=statsDir,speciesToRun=speciesToRun,beamSize=beamSize,targetBeamSize=targetBeamSize,Nspec=Nspec,bandwidths=bandwidths,output=output,Nsightlines1d=Nsightlines1d)
        sightline_indices = range(0,Nsightlines)
        print("Working with ",num_cores," threads!")
        x = Parallel(n_jobs=num_cores)(delayed(foo_)(i) for i in sightline_indices)
        
        print("SHAPE OF X=",np.shape(x))
        print("mean,max of x prior to summation ",np.mean(x),np.max(x))

        image = np.sum(x,0)
        print("mean,max of x after summation ",np.mean(x),np.max(x))

       # for i in sightline_indices:
       #    hf = h5py.File(output+
            
        
        hf = h5py.File(output+'_fullSpectra.hdf5','w')
        hf.create_dataset('spectra',data=image)
        hf.close()
        if savePNG:
            fig = plt.figure()
            vmax = np.max(np.sum(image,2))
            vmin = vmax * np.power(10.0,-4.0)
            plt.imshow(np.sum(image,2),norm=LogNorm(vmin=vmin,vmax=vmax),cmap='inferno')
            plt.colorbar()
            plt.savefig(output+'_fullSpectra.png')
            
        if bandwidth_km_s is not None:    
            Nx,Ny,Nspec = np.shape(image)
            dopplerVel = np.linspace(-bandwidth_km_s/2 , bandwidth_km_s/2,Nspec)
            dopplerMF = np.zeros((Nx,Ny))
            for i in range(0,Nx):
                for j in range(0,Ny):
                    dopplerMF[i,j] = np.sum(np.multiply(image[i,j,:] , dopplerVel)) #Analog to mass flux
            hf = h5py.File(output+".hdf5",'w')
            hf.create_dataset('image',data=dopplerMF)
            hf.close()
            if savePNG:
                fig = plt.figure()
                vmax = np.max([-np.min(dopplerMF),np.max(dopplerMF)])
                vmin = -vmax
                plt.imshow(dopplerMF,vmin=vmin,vmax=vmax,cmap='seismic')
                plt.colorbar()
                plt.savefig(output+'.png')
            
        
    else:
    	for i in range(0,Nsightlines):
            sightlineDir = outputs[0]+"_sightline"+str(i)+".hdf5"
            output = outputs+"_spectra"+str(i)+".hdf5"
            image[i,:]=IterateSightlines(snapDir,Nsnapstring,statsDir,output,sightlineDir,speciesToRun,beamSize,Nspec,bandwidths)
    print("Snapshot ",Nsnapstring," ran in ",time.time()-t1) 

    #return image
    
    
    
def ConvolveWithPSF(image,ix,iy,beamSize,targetBeamSize):
    #Project into 2D
    
    #CT: What should phi be? phi0? does it matter?
    
    #bin with appropriate resolution
    #For each bin convolve with psf(2-d gaussian kernel) and add result to modelImage 
    #lambda_k = .01#stdv in k direction #CT need to fit these with resolution of psf...
    #lambda_n = .01#stdv in n direction
    
    #lambda_k = beamSize * beam2pixel
    #lambda_n = beamSize * beam2pixel
    
    lambda_k = targetBeamSize / beamSize
    lambda_n = targetBeamSize / beamSize #way this is set up, should be about 1 px per beamsize...
    
    prefactor = 1/2/pi/lambda_k/lambda_n #lambda normalizaions cancel with beam resizing?
    
    newImage = np.copy(image)
    
    nx,ny,nspec = np.shape(image)
    
    gaussianMat = np.zeros((np.shape(image)[0]*2 , np.shape(image)[1]*2))
    
    
    x = np.indices(np.shape(gaussianMat))[0]
    y = np.indices(np.shape(gaussianMat))[1]
    
    xc = np.shape(gaussianMat)[0]/2
    yc = np.shape(gaussianMat)[1]/2
    
    gaussianMat = prefactor * np.exp(-0.5 * (np.power(x-xc,2)/lambda_k/lambda_k + np.power(y-yc,2)/lambda_n/lambda_n)) #add psf from pixel
    normCorrection = np.sum(gaussianMat)
    gaussianMat*=1/normCorrection #account for resolution limitations for a very small psf in this pixel space

    gaussianMat = np.repeat(gaussianMat[:,:,np.newaxis],nspec,axis=2)
    
    
    newImage = np.multiply(image[ix,iy,:] , gaussianMat[(nx-ix):(2*nx-ix),(ny-iy):(2*ny-iy),:])

    return newImage
    
    
def CreateSubImage(fullImageDir, subImageDir, i, j, k,inclination,NpixelNeighbors=2,NspectralNeighbors=10):
    hf_full = h5py.File(fullImageDir,'r')
    fullImage = np.array(hf_full['spectra'])
    hf_full.close()
    
    partial_image = np.zeros((NpixelNeighbors*2+1,NpixelNeighbors*2+1,NspectralNeighbors*2+1))
    partial_image[:,:,:] = fullImage[ (i-NpixelNeighbors):(i+NpixelNeighbors+1) ,(j-NpixelNeighbors):(j+NpixelNeighbors+1) ,k:k+NspectralNeighbors*2+1 ]
    hf_sub = h5py.File(subImageDir,'w')
    hf_sub.create_dataset('spectra',data=partial_image)
    hf_sub.create_dataset('nx',data=i)
    hf_sub.create_dataset('ny',data=j)
    hf_sub.create_dataset('ns',data=k)
    hf_sub.create_dataset('i',data=inclination)
    hf_sub.close()
    
    
    

