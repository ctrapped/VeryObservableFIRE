import numpy as np
pi=np.pi
arcsec = (1. /60. / 60.) * pi/180.
####Modify the values of each parameter to run VeryObservableFIRE. Pass the name of this file (e.g param_template) when you run VOF_CreateDataset.py
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
    sightlineDir=output+'sightlines\\'+galName #Subdirectory to store sightline files
    #############################

    return galName,minSnap,maxSnap,fileDir,statsDir,output,sightlineDir

def LoadObserverInfo(set_inclination=None):
    #### Observer parameters ####
    observerDistance=10000 #Distance in kpc
    observerVelocity=np.array([0,0,0]) #Observer velocity
    maxRadius=30 #max radius from disk center to image
    maxHeight=10 #max height above disk plane to include
    targetBeamSize=6*arcsec #beam size of instrument being modeled (in radians)
    Nsightlines1d=600 #number of sightlines along one axis
    phiObs=0 #offset image with this (radians)
    inclinations = np.array([30,40,50,60]) #Inclinations to image (degrees)
    position_angles = [0,45,90,135,180,225,270,315] #Position angles to image (degrees)

    speciesToRun='HI_21cm' #List of spectra to run
    bandwidth_km_s = 400. #bandwidth in km/s
    res_km_s = 5.2 #spectral resolution in km/s
    #############################
    
    if set_inclination is not None: inclinations=[set_inclination]
    
    return observerDistance,observerVelocity,maxRadius,maxHeight,targetBeamSize,Nsightlines1d,phiObs,inclinations,position_angles,speciesToRun,bandwidth_km_s,res_km_s

def LoadParameters():
    #### Run Parameters ####
    replaceAnnotationsFile=True #[False]=Append to existing annotation file. [True]=Overwrite existing annotation File
    runBinfire=True #[True]=Generate Annotation Files
    runVOF=False #[True]=Generate Spectral Datacubes

    createSightlineFiles=False #[True]=Create New sightline files
    savePng=True #[True]=Generate images showing annotations+images

    writeMassFlux=True #[True]=Generate mass flux annotations
    writeMass=True #[True]=Generate mass annotations
    writeRotationCurve=True #[True]=Generate rotation curve annotations
    writeInclination=True
    createMaskFromExistingStatsDir=False #Mask the previously run galaxy to find satellites/other galaxies in snapshot
    #############################

    return replaceAnnotationsFile,runBinfire,runVOF,createSightlineFiles,savePng,writeMassFlux,writeMass,writeRotationCurve,writeInclination,createMaskFromExistingStatsDir