import os
import numpy as np
from GalFit_ImageConverter_VelocitySpace import FireToDataset
from VeryObservableFIRE.VOF_EmissionSpecies import GetEmissionSpeciesParameters

pi = np.pi
arcsec = (1. /60. / 60.) * pi/180.
c_km_s = 3*10**5 #speed of light in km/s
h = 4.135667696*np.power(10.,-15.) #eV * s


galName="m12i"
#Snapshot Location#
filedir = '..\\FIRE_Simulations\\'+galName+'_cr700\\output\\snapdir_'
statsDir= '..\\FIRE_Simulations\\'+galName+'_cr700\\stats\\'+galName+'_cr700_stats_'
#output='data\\fire2_cr\\training\\'+galName+'_cr700_'
output='galfitData\\fire2_velocitySpace\\'
sightlineDir=output+'sightlines\\'+galName+'_cr700_'
rotationCurveDirectory = output+'sightlines\\'+galName+'_cr700_RC'
observerDistance=5000
observerVelocity=np.array([[0,0,0]]) ##Define as solar velocity??
maxRadius=20
maxHeight=4
binsizes = [2,2,8]
#binsizes=[2,2,8]
targetBeamSize=6*arcsec
#Nsightlines1d=6
Nsightlines1d=20
beamSize = 2*maxRadius/Nsightlines1d / observerDistance
phiObs=0
#inclinations=np.array([60])
#inclinations=np.array([30,40,50,60,70,80])
inclinations=np.array([50])

speciesToRun=['HI_21cm']

NpixelNeighbors = 2
NspectralNeighbors = 4

bandwidth_km_s = 400.
res_km_s = 5.2#EDIT sightline gen to sum emission bins or something to match res

#diskRadius=17.1
#diskHeight = 2



mass_species,g_upper,g_lower,E_upper,E_lower,A_ul,gamma_ul,Glevels,Elevels,n_u_fraction,n_l_fraction = GetEmissionSpeciesParameters(speciesToRun[0]) #MOVE THIS INTO GENERATESPECTRA
f0 = (E_upper-E_lower)/h #in Hz
Nspec = int(np.ceil(bandwidth_km_s / res_km_s))
bandwidth = [f0*c_km_s * (1 / (c_km_s-bandwidth_km_s/2) - 1 / (c_km_s+bandwidth_km_s/2))]



print("##################    Calculated Observation Parameters    ###############")
print("Beamsize = ",beamSize)
print("f0 = ",f0)
print("Bandwidth = ",bandwidth)
print("###########################################################################")
#Nspec=5000
#bandwidth=[5*10**6]



#FireToDataset(filedir,statsDir,Nsnap,output,sightlineDir,observerDistance,observerVelocity,maxRadius,maxHeight,binsizes,beamSize,Nsightlines1d,phiObs,inclinations,speciesToRun,Nspec,bandwidth,False,False)
replaceAnnotationsFile=True
runBinfire=True
runVOF=False

createSightlineFiles=True
savePng=True

writeMassFlux=True
writeMass=True
writeTiltedRing=False

#Nsnap = 600
#FireToDataset(filedir,statsDir,Nsnap,output,sightlineDir,galName,observerDistance,observerVelocity,maxRadius,maxHeight,binsizes,beamSize,Nsightlines1d,phiObs,inclinations,speciesToRun,Nspec,bandwidth,runBinfire,replaceAnnotationsFile,runVOF,savePng,writeMassFlux,writeMass)
    
for Nsnap in range(590,591):
    FireToDataset(filedir,statsDir,Nsnap,output,sightlineDir,galName,observerDistance,observerVelocity,maxRadius,maxHeight,binsizes,beamSize,targetBeamSize,Nsightlines1d,phiObs,inclinations,speciesToRun,Nspec,bandwidth,[bandwidth_km_s],runBinfire,replaceAnnotationsFile,runVOF,savePng,writeMassFlux,writeMass,writeTiltedRing,createSightlineFiles=createSightlineFiles)
    replaceAnnotationsFile=False
