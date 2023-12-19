import numpy as np
pi=np.pi
arcsec = (1. /60. / 60.) * pi/180.
####Modify the values of each parameter to run VeryObservableFIRE. Pass the name of this file (e.g param_template) when you run VOF_CreateDataset.py
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023

#### File Parameters ####
galName = 'm12i' #Simulation name as a string
minSnap = 581 #Starting snapshot number as an int
maxSnap = 590 #Ending snapshot number as an int
fileDir = '..\\FIRE_Simulations\\'+galName+'_cr700\\output\\snapdir_' #Path to the directory with snapshots. Should end without the trailing snapshot number
statsDir= '..\\FIRE_Simulations\\'+galName+'_cr700\\stats\\'+galName+'_cr700_stats_' #Path to a directory to store stats info. Will create .hdf5 file if doesn't exist
output= 'galfitData\\fire2_velocitySpace\\' #Directory to write outputs
sightlineDir=output+'sightlines\\'+galName+'_cr700' #Subdirectory to store sightline files
#############################


#### Run Parameters ####
replaceAnnotationsFile=False #[False]=Append to existing annotation file. [True]=Overwrite existing annotation File
runBinfire=True #[True]=Generate Annotation Files
runVOF=True #[True]=Generate Spectral Datacubes

createSightlineFiles=True #[True]=Create New sightline files
savePng=True #[True]=Generate images showing annotations+images

writeMassFlux=True #[True]=Generate mass flux annotations
writeMass=True #[True]=Generate mass annotations
writeRotationCurve=True #[True]=Generate rotation curve annotations
#############################


#### Observer parameters ####
observerDistance=5000 #Distance in kpc
observerVelocity=np.array([0,0,0]) #Observer velocity
maxRadius=40 #max radius from disk center to image
maxHeight=10 #max height above disk plane to include
targetBeamSize=6*arcsec #beam size of instrument being modeled (in radians)
Nsightlines1d=40 #number of sightlines along one axis
phiObs=0 #offset image with this (radians)
inclinations = np.array([40,50,60]) #Inclinations to image (degrees)

speciesToRun=['HI_21cm'] #List of spectra to run
bandwidth_km_s = 400. #bandwidth in km/s
res_km_s = 5.2 #spectral resolution in km/s
#############################


def LoadFileInfo(set_galName=None,set_minSnap=None,set_maxSnap=None):
    if set_galName is not None: galName=set_galName
    if set_minSnap is not None: minSnap=set_minSnap
    if set_maxSnap is not None: maxSnap=set_maxSnap

    return galName,minSnap,maxSnap,fileDir,statsDir,output,sightlineDir

def LoadObserverInfo(set_inclination=None):
    if set_inclination is not None: inclinations=[set_inclination]
    return observerDistance,observerVelocity,maxRadius,maxHeight,targetBeamSize,Nsightlines1d,phiObs,inclinations,speciesToRun,bandwidth_km_s,res_km_s

def LoadParameters():
    return replaceAnnotationsFile,runBinfire,runVOF,createSightlineFiles,savePng,writeMassFlux,writeMass,writeRotationCurve
    
    