import os
import numpy as np
import time
import sys
import importlib

from VOF_ImageConverter import FireToDataset
from VOF_EmissionSpecies import GetEmissionSpeciesParameters

####Runs VeryObservableFIRE to create synthetic images from given observational parameters.
####Also creates corresponding projected and deprojected maps of radial mass flux, rotational velocity, and mass for the purposes of neural network training
####To use, modify param_template.py and pass the renamed file as the first argument.
####    e.g. python VOF_CreateDataset.py param_template
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023



pi = np.pi
arcsec = (1. /60. / 60.) * pi/180.
c_km_s = 3*10**5 #speed of light in km/s
h = 4.135667696*np.power(10.,-15.) #eV * s
startTime=time.time()

paramFile = sys.argv[1]
paramMod = importlib.import_module(paramFile)

try:
    galName = sys.argv[2]
    minSnap = int(sys.argv[3])
    maxSnap = int(sys.argv[4])
    inclination = int(sys.argv[5])
except:
    galName=None
    minSnap=None
    maxSnap=None
    inclination=None

galName,minSnap,maxSnap,fileDir,statsDir,output,sightlineDir = paramMod.LoadFileInfo(galName,minSnap,maxSnap)

print("Looking at:"+fileDir)

observerDistance,observerVelocity,maxRadius,maxHeight,targetBeamSize,Nsightlines1d,phiObs,inclinations,speciesToRun,bandwidth_km_s,res_km_s=paramMod.LoadObserverInfo(inclination)

beamSize = 2*maxRadius/Nsightlines1d / observerDistance
noiseAmplitude = 0.0004 #Jy


mass_species,g_upper,g_lower,E_upper,E_lower,A_ul,gamma_ul,Glevels,Elevels,n_u_fraction,n_l_fraction = GetEmissionSpeciesParameters(speciesToRun)
f0 = (E_upper-E_lower)/h #in Hz
Nspec = int(np.ceil(bandwidth_km_s / res_km_s))
bandwidth = f0*c_km_s * (1 / (c_km_s-bandwidth_km_s/2) - 1 / (c_km_s+bandwidth_km_s/2))



print("##################    Calculated Observation Parameters    ###############")
print("Beamsize = ",beamSize)
print("f0 = ",f0)
print("Bandwidth = ",bandwidth)
print("###########################################################################")

replaceAnnotationsFile,runBinfire,runVOF,createSightlineFiles,savePng,writeMassFlux,writeMass,writeRotationCurve=paramMod.LoadParameters()

for Nsnap in range(minSnap,maxSnap+1):
    FireToDataset(fileDir,statsDir,Nsnap,output,sightlineDir,galName,observerDistance,observerVelocity,maxRadius,maxHeight,noiseAmplitude,beamSize,targetBeamSize,Nsightlines1d,phiObs,inclinations,speciesToRun,Nspec,bandwidth,bandwidth_km_s,runBinfire,replaceAnnotationsFile,runVOF,savePng,writeMassFlux,writeMass,writeRotationCurve,createSightlineFiles=createSightlineFiles)
    replaceAnnotationsFile=False
    
print("Time to finish: ",time.time()-startTime)
