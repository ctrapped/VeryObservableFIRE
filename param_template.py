import numpy as np
pi=np.pi
arcsec = (1. /60. / 60.) * pi/180.
####Modify the values of each parameter to run VeryObservableFIRE. Pass the name of this file (e.g param_template) when you run VeryObservableFIRE.py
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023

def LoadFileInfo(galName=None,minSnap=None,maxSnap=None):
    #### File Parameters ####
    if galName is None: galName = 'm12m' #Simulation name as a string
    if minSnap is None: minSnap = 600 #Starting snapshot number as an int
    if maxSnap is None: maxSnap = 600 #Ending snapshot number as an int
    fileDir = '/Volumes/wde4tb/simulation_snapshots/fire-2/'+galName+'/snapdir_' #Path to the directory with snapshots. Should end without the trailing snapshot number
    statsDir= '/Volumes/wde4tb/simulation_snapshots/fire-2/'+galName+'/stats/'+galName+"_stats" #Path to a directory to store stats info. Will create .hdf5 file if doesn't exist
    output= '/Volumes/wde4tb/simulation_snapshots/fire-2/'+galName+'/vof_outputs/' #Directory to write outputs
    #############################

    return galName,minSnap,maxSnap,fileDir,statsDir,output

def LoadObserverInfo(set_inclination=None):
    #### Observer parameters ####
    observerDistance=10000 #Distance in kpc
    observerVelocity=np.array([0,0,0]) #Observer velocity
    maxRadius=30 #max radius from disk center to image
    maxHeight=10 #max height above disk plane to include
    targetBeamSize=6*arcsec #beam size of instrument being modeled (in radians)
    Nsightlines1d=256 #number of sightlines along one axis
    phiObs=0 #offset image with this (radians)
    inclinations = np.array([30]) #Inclinations to image (degrees)
    position_angles = [0] #Position angles to image (degrees)

    speciesToRun='HI_21cm' #List of spectra to run
    res_km_s = 5.2 #spectral resolution in km/s
    bandwidth_km_s = res_km_s * 256 #bandwidth in km/s

    noiseAmplitude = 4e-4
    #############################
    
    if set_inclination is not None: inclinations=[set_inclination]
    
    return observerDistance,observerVelocity,maxRadius,maxHeight,targetBeamSize,Nsightlines1d,phiObs,inclinations,position_angles,speciesToRun,bandwidth_km_s,res_km_s,noiseAmplitude

def LoadParameters():
    #### Run Parameters ####
    replaceAnnotationsFile=True #[False]=Append to existing annotation file. [True]=Overwrite existing annotation File
    runBinfire=True #[True]=Generate Annotation Files
    runVOF=True #[True]=Generate Spectral Datacubes

    savePng=True #[True]=Generate images showing annotations+images

    writeMassFlux=True #[True]=Generate mass flux annotations
    writeMass=True #[True]=Generate mass annotations
    writeRotationCurve=True #[True]=Generate rotation curve annotations
    createMaskFromExistingStatsDir=False #Mask the previously run galaxy to find satellites/other galaxies in snapshot
    runDataAugmentation=True #[True]=Rotate/flip each generated image+annotations (VOF_ImageRotater.py) and append the augmented images to the annotation csvs. Requires runBinfire and runVOF to be True.
    num_cores = None #Set the number of cores to use in parallel sightline generation. Defaults to all available if None
    #############################

    return replaceAnnotationsFile,runBinfire,runVOF,savePng,writeMassFlux,writeMass,writeRotationCurve,createMaskFromExistingStatsDir,runDataAugmentation,num_cores