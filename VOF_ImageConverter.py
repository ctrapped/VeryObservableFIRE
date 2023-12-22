import numpy as np
import h5py

from VeryObservableFIRE import GenerateSyntheticImage
from Binfire.Binfire import RunBinfire

import matplotlib.pyplot as plt

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
                    writeMassFlux=True,writeMass=True,writeRotationCurve=True,
                    createSightlineFiles=True,
    ):


    #Create synthetic image using VOF like code
    #do for a variety of inclinations + in disk observations!
    if createAnnotations: #Run Binfire to create binned projection maps for radial mass flux, mass, and rotational velocities. Used as annotation files in NN training
        print("Creating Annotation files with Binfire...")
        for inclination in inclinations:
            image_name=output+"i"+str(inclination)+"/training/"+galName+"_cr700_i"+str(inclination)+"_"+str(Nsnap)+".hdf5"
            annotationFileDir_MF = output+"i"+str(inclination)+"/training/training_annotations_MassFlux_i"+str(inclination)
            annotationFileDir_Mass = output+"i"+str(inclination)+"/training/training_annotations_Mass_i"+str(inclination)
            annotationFileDir_RC = output+"i"+str(inclination)+"/training/training_annotations_RC_i"+str(inclination)


            binnedMass , binnedRadialMassFlux, binnedPhiMassFlux = RunBinfire(fileDir+str(Nsnap), 
                statsDir+str(Nsnap).zfill(4)+'.hdf5',
                Nsnap,
                output,
                [maxRadius,maxRadius,maxHeight],
                [Nsightlines1d,Nsightlines1d,Nspec],inclination=inclination
            )  


            if writeMassFlux:
                AppendToAnnotationsFile(annotationFileDir_MF+".csv",image_name,binnedRadialMassFlux.flatten(),replaceAnnotationsFile=replaceAnnotationsFile)
                if savePNG:
                    vmax = np.max([-np.min(binnedRadialMassFlux) , np.max(binnedRadialMassFlux)])
                    vmin=-vmax
                    plt.figure()
                    plt.imshow(binnedRadialMassFlux,vmin=vmin,vmax=vmax,cmap='seismic')
                    plt.savefig(annotationFileDir_MF+"_"+galName+"_RC_"+str(Nsnap)+".png")
                    plt.close()

            if writeMass:
                AppendToAnnotationsFile(annotationFileDir_Mass+".csv",image_name,binnedMass.flatten(),replaceAnnotationsFile=replaceAnnotationsFile)
                if savePNG:
                    vmax = np.max(binnedMass)
                    vmin=vmax*1e-4
                    plt.figure()
                    plt.imshow(binnedMass,vmin=vmin,vmax=vmax,cmap='inferno')
                    plt.savefig(annotationFileDir_Mass+"_"+galName+"_Mass_"+str(Nsnap)+".png")
                    plt.close()
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
                
    
    
    
    if createImages:
        print("Creating Synthetic Images...")
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