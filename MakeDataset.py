import numpy as np
import os
import copy
from GenerateSynetheticImage import GenerateSyntheticImage
from Binfire.Binfire import RunBinfire
from Binfire.Binfire import LoadGas
from Binfire.readsnap_binfire import ReadStats
from LoadData import LoadDataForImageGen
from LoadData import LoadMaskedDataForImageGen
from OrientGalaxy import OrientGalaxy
from WriteAnnotations import WriteAnnotations
import time

import scipy
####Function converts a FIRE snapshot to an image dataset usable with CoNNGaFit.
####Based on given options will first generate annotation files in the form of a .csv file for the mass flux, mass, and/or rotational velocities
####Will then generate synthetic images corresponding to those projection maps.
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023

def MakeDataset(config, fileDir, statsDir, Nsnap, output, galName, writeRadialVelocity=True):

    #Config values this function's own body needs. Values only needed by GenerateSyntheticImage/ProjectImage
    #(e.g. observerDistance, beamSize, num_cores, ...) are read directly out of config there instead.
    maxRadius = config['maxRadius']
    maxHeight = config['maxHeight']
    Nsightlines1d = config['Nsightlines1d']
    inclinations = config['inclinations']
    position_angles = config['position_angles']
    speciesToRun = config['speciesToRun']
    Nspec = config['Nspec']
    createAnnotations = config['runBinfire']
    createImages = config['runVOF']
    savePNG = config['savePng']
    createMaskFromExistingStatsDir = config['createMaskFromExistingStatsDir']
    projectGasProperties = config['projectGasProperties']

    #Create synthetic image using VOF like code
    #do for a variety of inclinations + in disk observations!
    outputSuffix=""
    particles=None

    if createAnnotations or projectGasProperties: #Run Binfire to create binned projection maps for radial mass flux, mass, and rotational velocities. Used as annotation files in NN training
        maskCenter=None;maskRadius=None
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
            
        particles = LoadGas( #Load gas particles
                fileDir+str(Nsnap),
                statsDir+str(Nsnap).zfill(4)+'.hdf5',
                Nsnap,
                [maxRadius,maxRadius,maxHeight],
                maskCenter=maskCenter,maskRadius=maskRadius
        )

        if projectGasProperties:
            Gmom_r, Gmom_s, Gmom_phi = RunBinfire(fileDir+str(Nsnap),
                statsDir+str(Nsnap).zfill(4)+'.hdf5',copy.deepcopy(particles),
                Nsnap,
                output,
                [maxRadius,maxRadius,maxHeight],
                [Nsightlines1d,Nsightlines1d,Nspec],inclination=inclinations[0],position_angle=position_angles[0],
                maskCenter=maskCenter,maskRadius=maskRadius,project_gas_properties=projectGasProperties
            )
    
        else:
            print("Creating Annotation files with Binfire...")
            for inclination in inclinations:
                os.makedirs(output+"i"+str(inclination)+"/", exist_ok=True)
                for position_angle in position_angles:
                    binnedMass , binnedRadialMassFlux, binnedPhiMassFlux, binnedCylRadMassFlux, binnedCylRadMassFluxCurve, binnedInclination = RunBinfire(fileDir+str(Nsnap), 
                            statsDir+str(Nsnap).zfill(4)+'.hdf5',copy.deepcopy(particles),
                            Nsnap,
                            output,
                            [maxRadius,maxRadius,maxHeight],
                            [Nsightlines1d,Nsightlines1d,Nspec],inclination=inclination,position_angle=position_angle,
                            maskCenter=maskCenter,maskRadius=maskRadius
                    )  

                    WriteAnnotations(output, galName, inclination, position_angle, Nsnap, outputSuffix,
                                      binnedMass, binnedRadialMassFlux, binnedPhiMassFlux, binnedCylRadMassFlux, binnedCylRadMassFluxCurve,
                                      writeRadialVelocity=writeRadialVelocity,
                                      savePNG=savePNG)
        
    if createImages:
        print("Creating Synthetic Images...")

        Nsnapstring = str(Nsnap)
        r0,pos_center,Lhat,vel_center = ReadStats(statsDir+Nsnapstring.zfill(4)+'.hdf5')
        snapdir = fileDir+Nsnapstring #Directory containing the actual snapshots

        print("snapdir=",snapdir)

        #Load the gas particles if not loaded previously
        gPos,gKernel,gVel = LoadDataForImageGen(snapdir,Nsnapstring,0,maxRadius,pos_center,vel_center,particles=particles)

        if particles is None:
            rmag = np.linalg.norm(gPos,axis=1)
            radMask = np.where(rmag<maxRadius*1.5)[0]
            gPos = gPos[radMask]
            gKernel = gKernel[radMask]
            gVel = gVel[radMask]
            del rmag
        else:
            radMask = None
    
        #Load masked data if not loaded previously
        gMas,gTemp,speciesMassFrac = LoadMaskedDataForImageGen(snapdir,Nsnapstring,ptype=0,mask=radMask,gKernel=gKernel,species=speciesToRun,particles=particles)

        for inclination in inclinations:
          os.makedirs(output+"i"+str(inclination)+"/", exist_ok=True)
          for position_angle in position_angles:
            print("Generating Image for inclination: ",inclination)
            image_name=output+"i"+str(inclination)+"/"+galName+"_i"+str(inclination)+"_pa"+str(position_angle)+"_"+str(Nsnap)+"_image"+outputSuffix
            if os.path.isfile(image_name+"_fullSpectra.hdf5"):
                print("Warning: image for i=",inclination,"pa=",position_angle,"already exists. Overwriting...")
                
            if inclination>0:
                #Rotate r0 first to set the position angle
                rotation_axis = np.copy(Lhat)
                rotation_vector = float(position_angle)*np.pi/180.*rotation_axis
                rotation = scipy.spatial.transform.Rotation.from_rotvec(rotation_vector)
                r0 = rotation.apply(r0)
        
                #Rotate the vectors Lhat and r0 that define the z and x unit vectors respectively. Effectively rotates the entire galaxy
                rotation_axis = np.cross(Lhat,r0)
                rotation_vector = float(inclination)*np.pi/180.*rotation_axis
                rotation = scipy.spatial.transform.Rotation.from_rotvec(rotation_vector)

                Lhat_rot = rotation.apply(Lhat)
                r0_rot =   rotation.apply(r0)
            else:
                Lhat_rot = np.copy(Lhat)
                r0_rot = np.copy(r0)

            gPos_rot,gVel_rot = OrientGalaxy(copy.deepcopy(gPos),copy.deepcopy(gVel),Lhat_rot,r0_rot)

            particleData = {}
            particleData['pos'] = gPos_rot
            particleData['vel'] = gVel_rot
            particleData['kernel'] = copy.deepcopy(gKernel)
            particleData['mass'] = copy.deepcopy(gMas)
            particleData['temp'] = copy.deepcopy(gTemp)
            particleData['speciesMassFrac'] = copy.deepcopy(speciesMassFrac)

            if projectGasProperties:
                particleData['rMom'] = copy.deepcopy(Gmom_r)
                particleData['sMom'] = copy.deepcopy(Gmom_s)
                particleData['rotMom'] = copy.deepcopy(Gmom_phi)

            GenerateSyntheticImage(config, fileDir, statsDir, Nsnap, image_name, inclination, position_angle, particleData=particleData)