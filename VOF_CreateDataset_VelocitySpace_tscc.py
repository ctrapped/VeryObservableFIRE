import os
import numpy as np
import time
import sys

from VOF_ImageConverter_VelocitySpace import FireToDataset
from VOF_EmissionSpecies import GetEmissionSpeciesParameters

pi = np.pi
arcsec = (1. /60. / 60.) * pi/180.
c_km_s = 3*10**5 #speed of light in km/s
h = 4.135667696*np.power(10.,-15.) #eV * s
startTime=time.time()



paramFile = sys.argv[1]
paramMod = importlib.import_module(paramFile)

galName,minSnap,maxSnap,fileDir,statsDir,output,sightlineDir = paramMod.LoadFileInfo()

print("Looking at:"+filedir)

observerDistance,observerVelocity,maxRadius,maxHeight,binsizes,targetBeamSize,Nsightlines1d,phiObs,inclinations,speciesToRun,bandwidth_km_s,res_km_s=paramMod.LoadObserverInfo()

beamSize = 2*maxRadius/Nsightlines1d / observerDistance


mass_species,g_upper,g_lower,E_upper,E_lower,A_ul,gamma_ul,Glevels,Elevels,n_u_fraction,n_l_fraction = GetEmissionSpeciesParameters(speciesToRun[0]) #MOVE THIS INTO GENERATESPECTRA
f0 = (E_upper-E_lower)/h #in Hz
Nspec = int(np.ceil(bandwidth_km_s / res_km_s))
bandwidth = [f0*c_km_s * (1 / (c_km_s-bandwidth_km_s/2) - 1 / (c_km_s+bandwidth_km_s/2))]



print("##################    Calculated Observation Parameters    ###############")
print("Beamsize = ",beamSize)
print("f0 = ",f0)
print("Bandwidth = ",bandwidth)
print("###########################################################################")

replaceAnnotationsFile,runBinfire,runVOF,createSightlineFiles,savePng,writeMassFlux,writeMass,writeTiltedRing,writeRotationCurve=paramMod.LoadParameters()

#Nsnap = 600
#FireToDataset(filedir,statsDir,Nsnap,output,sightlineDir,galName,observerDistance,observerVelocity,maxRadius,maxHeight,binsizes,beamSize,Nsightlines1d,phiObs,inclinations,speciesToRun,Nspec,bandwidth,runBinfire,replaceAnnotationsFile,runVOF,savePng,writeMassFlux,writeMass)
    
for Nsnap in range(minSnap,maxSnap+1):
    FireToDataset(filedir,statsDir,Nsnap,output,sightlineDir,galName,observerDistance,observerVelocity,maxRadius,maxHeight,binsizes,beamSize,targetBeamSize,Nsightlines1d,phiObs,inclinations,speciesToRun,Nspec,bandwidth,[bandwidth_km_s],runBinfire,replaceAnnotationsFile,runVOF,savePng,writeMassFlux,writeMass,writeTiltedRing,writeRotationCurve,createSightlineFiles=createSightlineFiles)
    replaceAnnotationsFile=False
    
print("Time to completely finish 1 snapshot: ",time.time()-startTime)
