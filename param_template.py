import numpy as np
arcsec = (1. /60. / 60.) * pi/180.

#### File Parameters ####
galName = 'm12i' #Simulation name as a string
minSnap = 581 #Starting snapshot number as an int
maxSnap = 590 #Ending snapshot number as an int
fileDir = '..\\FIRE_Simulations\\'+galName+'_cr700\\output\\snapdir_' #Path to the directory with snapshots. Should end without the trailing snapshot number
statsDir= '..\\FIRE_Simulations\\'+galName+'_cr700\\stats\\'+galName+'_cr700_stats_' #Path to a directory to store stats info. Will create .hdf5 file if doesn't exist
output= 'galfitData\\fire2_velocitySpace\\' #Directory to write outputs
sightlineDir=output+'sightlines\\'+galName+'_cr700_' #Subdirectory to store sightline files
#############################


#### Run Parameters ####
replaceAnnotationsFile=False #[False]=Append to existing annotation file. [True]=Overwrite existing annotation File
runBinfire=True #[True]=Generate Annotation Files
runVOF=True #[True]=Generate Spectral Datacubes

createSightlineFiles=True #[True]=Create New sightline files
savePng=True #[True]=Generate images showing annotations+images

writeMassFlux=True #[True]=Generate mass flux annotations
writeMass=True #[True]=Generate mass annotations
writeTiltedRing=False #[True]=Generate tilted ring annotations [deprecated]
writeRotationCurve=True #[True]=Generate rotation curve annotations
#############################


#### Observer parameters ####
observerDistance=5000 #Distance in kpc
observerVelocity=np.array([0,0,0]) #Observer velocity
maxRadius=40 #max radius from disk center to image
maxHeight=10 #max height above disk plane to include
binsizes=[2,2,20] # spatial resolution of annotation data when run in de-projected mode
targetBeamSize=6*arcsec #beam size of instrument being modeled (in radians)
Nsightlines1d=40 #number of sightlines along one axis
phiObs=0 #offset image with this (radians)
inclinations = np.array([40,50,60]) #Inclinations to image (degrees)

speciesToRun=['HI_21cm'] #List of spectra to run
bandwidth_km_s = 400. #bandwidth in km/s
res_km_s = 5.2 #spectral resolution in km/s
#############################


def LoadFileInfo():
    return galName,minSnap,maxSnap,fileDir,statsDir,output,sightlineDir

def LoadObserverInfo():
    return observerDistance,observerVelocity,maxRadius,maxHeight,binsizes,targetBeamSize,Nsightlines1d,phiObs,inclinations,speciesToRun,bandwidth_km_s,res_km_s

def LoadParameters():
    return replaceAnnotationsFile,runBinfire,runVOF,createSightlineFiles,savePng,writeMassFlux,writeMass,writeTiltedRing,writeRotationCurve
    
    