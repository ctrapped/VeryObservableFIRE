import os
import pandas as pd
import numpy as np
import h5py
#from torchvision.io import read_image

from VeryObservableFIRE.VeryObservableFIRE import GenerateSyntheticImage
from Binfire.Binfire_VelocitySpace import RunBinfire

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def FireToDataset(fileDir,statsDir, Nsnap, output,sightlineDir,galName,
                    observerDistance, observerVelocity,
                    maxRadius,maxHeight,binsizes,
                    beamSize,targetBeamSize,Nsightlines1d,
                    phiObs,inclinations,
                    speciesToRun,Nspec,bandwidths,bandwidths_km_s,
                    createAnnotations=True,replaceAnnotationsFile=False,
                    createImages=True,savePNG=False,
                    writeMassFlux=True,writeMass=True,writeTiltedRing=True,writeRotationCurve=True,
                    createSightlineFiles=True,
    ):
    print("createImages=",createImages)
    if createAnnotations: 
    #    print("Running binfire...")
    #    print("Looking at "+fileDir+str(Nsnap))
   #     if writeTiltedRing:
   #       binnedMass , binnedRadialMassFlux,tiltedRingParameters = RunBinfire(fileDir+str(Nsnap), # Need to fix centering in this version?
   #         statsDir+str(Nsnap).zfill(4)+'.hdf5',
   #         Nsnap,
    #        output,
    #        [maxRadius,maxRadius,bandwidths_km_s[0]],
    #        binsizes,returnTiltedRingData=writeTiltedRing,Nspec=Nspec
    #      )  
    #    else:
    #      binnedMass , binnedRadialMassFlux = RunBinfire(fileDir+str(Nsnap), # Need to fix centering in this version?
    #        statsDir+str(Nsnap).zfill(4)+'.hdf5',
    #        Nsnap,
    #        output,
    #        [maxRadius,maxRadius,bandwidths_km_s[0]],
    #        binsizes,Nspec=Nspec
    #      )  
        
        #print("Shape of binnedRadialMassFlux=",np.shape(binnedRadialMassFlux))
         
        #print("Appending binfire to annotations file...")
        #annotationFileDir = output+"_annotations.csv"
        for inclination in inclinations:
            image_name=output+"i"+str(inclination)+"/training/"+galName+"_cr700_i"+str(inclination)+"_"+str(Nsnap)+"_10022023.hdf5"
            annotationFileDir_MF = output+"i"+str(inclination)+"/training/training_annotations_MassFluxTest_i"+str(inclination)+"_10022023.csv"
            annotationFileDir_Mass = output+"i"+str(inclination)+"/training/training_annotations_MassTest_i"+str(inclination)+"_10022023.csv"
            annotationFileDir_TR = output+"i"+str(inclination)+"/training/training_annotations_TiltedRing_i"+str(inclination)+"_10022023.csv"
            annotationFileDir_RC = output+"i"+str(inclination)+"/training/training_annotations_RCTest_i"+str(inclination)+"_10022023.csv"


            if writeTiltedRing:
                binnedMass , binnedRadialMassFlux,tiltedRingParameters = RunBinfire(fileDir+str(Nsnap), # Need to fix centering in this version?
                    statsDir+str(Nsnap).zfill(4)+'.hdf5',
                    Nsnap,
                    output,
                    [maxRadius,maxRadius,maxHeight],
                    [Nsightlines1d,Nsightlines1d,Nspec],returnTiltedRingData=writeTiltedRing,inclination=inclination
                )  
            else:
                binnedMass , binnedRadialMassFlux, binnedPhiMassFlux = RunBinfire(fileDir+str(Nsnap), # Need to fix centering in this version?
                    statsDir+str(Nsnap).zfill(4)+'.hdf5',
                    Nsnap,
                    output,
                    [maxRadius,maxRadius,maxHeight],
                    [Nsightlines1d,Nsightlines1d,Nspec],inclination=inclination
                )  



            if writeMassFlux:
                AppendToAnnotationsFile(annotationFileDir_MF,image_name,binnedMass=None,binnedRadialMassFlux=binnedRadialMassFlux.flatten(),replaceAnnotationsFile=replaceAnnotationsFile)
                plt.figure()
                vmax = np.max([-np.min(binnedRadialMassFlux) , np.max(binnedRadialMassFlux)])
                vmin=-vmax
                plt.imshow(binnedRadialMassFlux,vmin=vmin,vmax=vmax,cmap='seismic')
                plt.savefig(annotationFileDir_Mass+"_"+galName+"_RadMF_"+str(Nsnap)+".png")
                plt.close()
            if writeMass:
                AppendToAnnotationsFile(annotationFileDir_Mass,image_name,binnedMass=binnedMass.flatten(),binnedRadialMassFlux=None,replaceAnnotationsFile=replaceAnnotationsFile)
                vmax = np.max(binnedMass)
                vmin=vmax*1e-4
                plt.figure()
                plt.imshow(binnedMass,vmin=vmin,vmax=vmax,cmap='inferno')
                plt.savefig(annotationFileDir_Mass+"_"+galName+"_Mass_"+str(Nsnap)+".png")
                plt.close()
            if writeRotationCurve:
                binnedMass[binnedMass==0]=1e-10
                AppendToAnnotationsFile(annotationFileDir_RC,image_name,binnedMass=np.divide(binnedPhiMassFlux,binnedMass).flatten(),binnedRadialMassFlux=None,replaceAnnotationsFile=replaceAnnotationsFile)
                vmax = np.abs(np.max(np.divide(binnedPhiMassFlux,binnedMass)))
                vmin=0
                plt.figure()
                plt.imshow(np.divide(binnedPhiMassFlux,binnedMass),vmin=vmin,vmax=vmax,cmap='inferno')
                plt.savefig(annotationFileDir_RC+"_"+galName+"_RC_"+str(Nsnap)+".png")
                plt.close()
                
            if writeTiltedRing:
                AppendToAnnotationsFile(annotationFileDir_TR,image_name,tiltedRingParameters.flatten().flatten(),binnedRadialMassFlux=None,replaceAnnotationsFile=replaceAnnotationsFile)



    #Create synthetic image using VOF like code
    #do for a variety of inclinations + in disk observations!
    if createImages:
        for inclination in inclinations:
            print("Generating Image for inclination: ",inclination)
            image_name=output+"i"+str(inclination)+"/training/"+galName+"_cr700_i"+str(inclination)+"_"+str(Nsnap)+"_image_04172023"
            GenerateSyntheticImage(fileDir,
                statsDir, #If not provided, generate
                Nsnap,
                image_name,
                sightlineDir+"_i"+str(inclination)+"_"+str(Nsnap)+"_image",
                observerDistance,
                observerVelocity,
                maxRadius,
                beamSize,targetBeamSize,
                Nsightlines1d,
                phiObs,
                inclination,
                speciesToRun,
                Nspec,
                bandwidths,
                savePNG,createSightlineFiles,bandwidth_km_s=bandwidths_km_s[0]
                )
            
            
        #hf = h5py.File(output+"_i"+str(inclination)+"_image.hdf5",'w')
        #hf.create_dataset("spectra",data=image)
        #hf.close()
        
        #if savePNG:
        #    try:
        #        plt.imshow(np.sum(image,2),norm=LogNorm(),cmap='inferno')
        #    except:
        #        plt.imshow(np.sum(image,2),cmap='inferno')
        #    
        #    plt.savefig(output+"_i"+str(inclination)+"_image.png")

    
    
    
    #Create bins of mass and radial velocity using binfire like code and save as annotation csv file
    #maybe start with limiting to ~1kpc voxels within the disk? ~5 kpc




#def FitsToDataset(filedir,output):
    #Convert observational fits files to dataset
    
def AppendToAnnotationsFile(annotationFileDir,image_name,binnedMass=None,binnedRadialMassFlux=None,replaceAnnotationsFile=False):
    if replaceAnnotationsFile:
        fid = open(annotationFileDir,'w')
    else:
        try:
            fid = open(annotationFileDir,'a')
        except:
            print("Warning: Could not find existing Annotation File. Writing a new one...")
            fid = open(annotationFileDir,'w') 

    fid.write('\n'+image_name)
    if binnedMass is not None:
        for i in range(0,np.size(binnedMass)):
            fid.write(','+str(binnedMass[i]))

            
            
    if binnedRadialMassFlux is not None:
        for i in range(0,np.size(binnedRadialMassFlux)):
            fid.write(','+str(binnedRadialMassFlux[i]))
            
    fid.close()
    

    
            
def GetRadMfMap(binnedMass,bandwidth_km_s):
    Nx,Ny,Nspec = np.shape(binnedMass)
    rVel = np.linspace(-bandwidth_km_s/2 , bandwidth_km_s/2,Nspec)
    radMF = np.zeros((Nx,Ny))
    for i in range(0,Nx):
        for j in range(0,Ny):
            radMF[i,j] = np.sum(np.multiply(binnedMass[i,j,:] , rVel))
            
    return radMF
        
        
        
        
        
        
        
        
        
        
        
        
    