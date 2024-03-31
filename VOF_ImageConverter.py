import numpy as np
import h5py

from VeryObservableFIRE import GenerateSyntheticImage
from Binfire.Binfire import RunBinfire
from Binfire.readsnap_binfire import ReadStats

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
####Function converts a FIRE snapshot to a dataset usable with CoNNGaFit.
####Based on given options will first generate annotation files in the form of a .csv file for the mass flux, mass, and/or rotational velocities
####Will then generate synthetic images corresponding to those projection maps.
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023

def FireToDataset(fileDir,statsDir, Nsnap, output,sightlineDir,galName,
                    observerDistance, observerVelocity,
                    maxRadius,maxHeight,
                    noiseAmplitude,beamSize,targetBeamSize,Nsightlines1d,
                    phiObs,inclinations,
                    speciesToRun,Nspec,bandwidth,bandwidth_km_s,
                    createAnnotations=True,replaceAnnotationsFile=False,
                    createImages=True,savePNG=False,
                    writeMassFlux=True,writeMass=True,writeRotationCurve=True,writeRadialVelocity=True,writeInclination=True,
                    createSightlineFiles=True,
                    createMaskFromExistingStatsDir=False,
    ):


    #Create synthetic image using VOF like code
    #do for a variety of inclinations + in disk observations!
    if createAnnotations: #Run Binfire to create binned projection maps for radial mass flux, mass, and rotational velocities. Used as annotation files in NN training
        print("Creating Annotation files with Binfire...")
        
        maskCenter=None;maskRadius=None
        outputSuffix=""
        if createMaskFromExistingStatsDir:
            try:
                r_0,pos_center,Lhat,vel_center = ReadStats(statsDir+str(Nsnap).zfill(4)+'.hdf5')
                maskCenter = pos_center
                maskRadius = 100
                statsDir += "masked_"
                outputSuffix="_masked"
                #output += "masked_"
                print("Set mask center...")
            except:
                print("Warning, could not mask data as no previous stats file exists...")
            
        for inclination in inclinations:
            image_name=output+"i"+str(inclination)+"/training/"+galName+"_cr700_i"+str(inclination)+"_"+str(Nsnap)+outputSuffix+".hdf5"
            annotationFileDir_MF = output+"i"+str(inclination)+"/training/training_annotations_MassFlux_i"+str(inclination)+outputSuffix
            annotationFileDir_Mass = output+"i"+str(inclination)+"/training/training_annotations_Mass_i"+str(inclination)+outputSuffix
            annotationFileDir_RC = output+"i"+str(inclination)+"/training/training_annotations_RC_i"+str(inclination)+outputSuffix
            annotationFileDir_rVel = output+"i"+str(inclination)+"/training/training_annotations_rVel_i"+str(inclination)+outputSuffix
            annotationFileDir_sMF = output+"i"+str(inclination)+"/training/training_annotations_sMassFlux_i"+str(inclination)+outputSuffix
            annotationFileDir_sMF1d = output+"i"+str(inclination)+"/training/training_annotations_sMassFluxCurve_i"+str(inclination)+outputSuffix
            annotationFileDir_Inclination = output+"i"+str(inclination)+"/training/training_annotations_inclination_i"+str(inclination)+outputSuffix


            binnedMass , binnedRadialMassFlux, binnedPhiMassFlux, binnedCylRadMassFlux, binnedCylRadMassFluxCurve, binnedInclination = RunBinfire(fileDir+str(Nsnap), 
                statsDir+str(Nsnap).zfill(4)+'.hdf5',
                Nsnap,
                output,
                [maxRadius,maxRadius,maxHeight],
                [Nsightlines1d,Nsightlines1d,Nspec],inclination=inclination,
                maskCenter=maskCenter,maskRadius=maskRadius
            )  


            if writeMassFlux:
                AppendToAnnotationsFile(annotationFileDir_MF+".csv",image_name,binnedRadialMassFlux.flatten(),replaceAnnotationsFile=replaceAnnotationsFile)
                if savePNG:
                    vmax = np.max([-np.min(binnedRadialMassFlux) , np.max(binnedRadialMassFlux)])*.5
                    vmin=-vmax
                    plt.figure()
                    plt.imshow(binnedRadialMassFlux,vmin=vmin,vmax=vmax,cmap='seismic')
                    plt.colorbar()
                    plt.savefig(annotationFileDir_MF+"_"+galName+"_MF_"+str(Nsnap)+".png")
                    plt.close()
                    
                hfMF=h5py.File(annotationFileDir_MF+"_"+galName+"_MF_"+str(Nsnap)+".hdf5",'w')
                hfMF.create_dataset('imageName',data=image_name)
                hfMF.create_dataset('annotation',data=binnedRadialMassFlux.flatten())
                hfMF.close()
 
 
                AppendToAnnotationsFile(annotationFileDir_sMF+".csv",image_name,binnedCylRadMassFlux.flatten(),replaceAnnotationsFile=replaceAnnotationsFile)
                if savePNG:
                    vmax = np.max([-np.min(binnedCylRadMassFlux) , np.max(binnedCylRadMassFlux)])*.5
                    vmin=-vmax
                    plt.figure()
                    plt.imshow(binnedCylRadMassFlux,vmin=vmin,vmax=vmax,cmap='seismic')
                    plt.colorbar()
                    plt.savefig(annotationFileDir_sMF+"_"+galName+"_sMF_"+str(Nsnap)+".png")
                    plt.close()
                    
                hfsMF=h5py.File(annotationFileDir_sMF+"_"+galName+"_sMF_"+str(Nsnap)+".hdf5",'w')
                hfsMF.create_dataset('imageName',data=image_name)
                hfsMF.create_dataset('annotation',data=binnedCylRadMassFlux.flatten())
                hfsMF.close()
                
                hfsMF1d=h5py.File(annotationFileDir_sMF1d+"_"+galName+"_sMF1d_"+str(Nsnap)+".hdf5",'w')
                hfsMF1d.create_dataset('imageName',data=image_name)
                hfsMF1d.create_dataset('annotation',data=binnedCylRadMassFluxCurve.flatten())
                hfsMF1d.close()
                
                
                
                
            if writeRadialVelocity:
                binnedMass[binnedMass==0]=1e-10
                AppendToAnnotationsFile(annotationFileDir_rVel+".csv",image_name,np.divide(binnedCylRadMassFlux,binnedMass).flatten(),replaceAnnotationsFile=replaceAnnotationsFile)
                if savePNG:
                    vmax = np.max([-np.min(np.divide(binnedCylRadMassFlux,binnedMass)) , np.max(np.divide(binnedCylRadMassFlux,binnedMass))])
                    vmin=-vmax
                    plt.figure()
                    plt.imshow(np.divide(binnedCylRadMassFlux,binnedMass),vmin=vmin,vmax=vmax,cmap='seismic')
                    plt.colorbar()
                    plt.savefig(annotationFileDir_rVel+"_"+galName+"_rVel_"+str(Nsnap)+".png")
                    plt.close()
                    
                hfrVel=h5py.File(annotationFileDir_rVel+"_"+galName+"_rVel_"+str(Nsnap)+".hdf5",'w')
                hfrVel.create_dataset('imageName',data=image_name)
                hfrVel.create_dataset('annotation',data=np.divide(binnedCylRadMassFlux,binnedMass).flatten())
                hfrVel.close()

            if writeMass:
                AppendToAnnotationsFile(annotationFileDir_Mass+".csv",image_name,binnedMass.flatten(),replaceAnnotationsFile=replaceAnnotationsFile)
                if savePNG:
                    vmax = np.max(binnedMass)
                    vmin=vmax*1e-3
                    plt.figure()
                    plt.imshow(binnedMass,cmap='inferno',norm=LogNorm(vmin=vmin,vmax=vmax))
                    plt.savefig(annotationFileDir_Mass+"_"+galName+"_Mass_"+str(Nsnap)+".png")
                    plt.close()
                    
                hfMass=h5py.File(annotationFileDir_Mass+"_"+galName+"_Mass_"+str(Nsnap)+".hdf5",'w')
                hfMass.create_dataset('imageName',data=image_name)
                hfMass.create_dataset('annotation',data=binnedMass.flatten())
                hfMass.close()
                
            if writeRotationCurve:
                binnedMass[binnedMass==0]=1e-10
                AppendToAnnotationsFile(annotationFileDir_RC+".csv",image_name,np.divide(binnedPhiMassFlux,binnedMass).flatten(),replaceAnnotationsFile=replaceAnnotationsFile)
                if savePNG:
                    vmax = np.abs(np.max(np.divide(binnedPhiMassFlux,binnedMass)))
                    vmin=0
                    plt.figure()
                    plt.imshow(np.divide(binnedPhiMassFlux,binnedMass),vmin=vmin,vmax=vmax,cmap='inferno')
                    plt.savefig(annotationFileDir_RC+"_"+galName+"_RC_"+str(Nsnap)+".png")
                    plt.close()
                hfMass=h5py.File(annotationFileDir_RC+"_"+galName+"_RC_"+str(Nsnap)+".hdf5",'w')
                hfMass.create_dataset('imageName',data=image_name)
                hfMass.create_dataset('annotation',data=np.divide(binnedPhiMassFlux,binnedMass).flatten())
                hfMass.close()
                
            if writeInclination:
                binnedMass[binnedMass==0]=1e-10
                AppendToAnnotationsFile(annotationFileDir_Inclination+".csv",image_name,np.divide(binnedPhiMassFlux,binnedMass).flatten(),replaceAnnotationsFile=replaceAnnotationsFile)
                if savePNG:
                    vmax = inclination+20
                    vmin= inclination-20
                    plt.figure()
                    plt.imshow(binnedInclination,vmin=vmin,vmax=vmax,cmap='seismic')
                    plt.colorbar()
                    plt.savefig(annotationFileDir_Inclination+"_"+galName+"_inc_"+str(Nsnap)+".png")
                  
                    plt.close()
                hfMass=h5py.File(annotationFileDir_Inclination+"_"+galName+"_inc_"+str(Nsnap)+".hdf5",'w')
                hfMass.create_dataset('imageName',data=image_name)
                hfMass.create_dataset('annotation',data=binnedInclination.flatten())
                hfMass.close()
    
    
    if createImages:
        print("Creating Synthetic Images...")
        for inclination in inclinations:
            print("Generating Image for inclination: ",inclination)
            image_name=output+"i"+str(inclination)+"/training/"+galName+"_cr700_i"+str(inclination)+"_"+str(Nsnap)+"_image_04172023"+outputSuffix
            GenerateSyntheticImage(fileDir,
                statsDir, #If not provided, generate
                Nsnap,
                image_name,
                sightlineDir+"_i"+str(inclination)+"_"+str(Nsnap)+"_image",
                observerDistance,
                observerVelocity,
                maxRadius,
                noiseAmplitude,beamSize,targetBeamSize,
                Nsightlines1d,
                phiObs,
                inclination,
                speciesToRun,
                Nspec,
                bandwidth,
                savePNG,createSightlineFiles,bandwidth_km_s=bandwidth_km_s
                )





            
                
def AppendToAnnotationsFile(annotationFileDir,image_name,binnedData,replaceAnnotationsFile=False):
    #Append the results of Binfire to the next line of the .csv annotation file
    if replaceAnnotationsFile:
        fid = open(annotationFileDir,'w')
    else:
        try:
            fid = open(annotationFileDir,'a')
        except:
            print("Warning: Could not find existing Annotation File. Writing a new one...")
            fid = open(annotationFileDir,'w') 

    fid.write('\n'+image_name)
    
    for i in range(0,np.size(binnedData)):
        fid.write(','+str(binnedData[i]))

    fid.close()