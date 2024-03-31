import numpy as np
import math
import sys
import time
from VOF_GenerateSpectra import GenerateSpectra

unit_M = 10**10 * 1.98855 *10**33 #10^10 solar masses / h !!in grams!! #h accounted for in readsnap
unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
unit_V = 1.0*10.0**5 # 1 km/s !!in cm/s!! Converted by factor of sqrt(a) in readsnap
unit_T = unit_L/unit_V
unit_rho = unit_M / unit_L**3

proton_mass = 1.6726219*10**(-27)*(1000.0/unit_M) ##appropriate units

pi = np.pi
eps = 1e-10

####Function to find all particles that intersect with a given sightline of finite beam size
####Sorts particles by distance, such that the full spectrum can be built up by accounting for emission then absorption starting with the furthest particle
####
####Calls GenerateSpectra to determine the emission and full spectra from all particles in the given sightline
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 03-10-2023


def GetParticlesInSightline(sightline,testPositions,testVel,testRadii,testMass,testTemp,testSpeciesMassFrac,beamSize,speciesToRun,Nspec,bandwidth,rObserver,calcThermalLevels): #assume already centered on observer
    #print("Setting up sightline geometry...")
    #radMask = np.where(rmag<max_r*1.5)[0]
    zHat = sightline / np.linalg.norm(sightline) #orient cylindrical coordinate system along the sightline
    N,dim = np.shape(testPositions)
    r_z = np.zeros((N,3)) #vector for the z component in the old coord system

    zmag = np.zeros((N)) 
    zmag = np.dot(testPositions,zHat)
    r_z[:,0] = zmag*zHat[0]
    r_z[:,1] = zmag*zHat[1]
    r_z[:,2] = zmag*zHat[2]
    
    r_s=np.zeros((N,3))
    r_s[:,:] = np.subtract(testPositions[:,:],r_z[:,:]) #vector for the s component in the old coord system
    del testPositions
    del r_z
    smag=np.zeros((N))
    smag[:] = np.linalg.norm(r_s,axis=1)
    del r_s
   
    beamRadiusPhysical=np.zeros((N))
    beamRadiusPhysical[:] = zmag[:] * beamSize / 2 #the physical size of the beam for each particle
    
   
    mask = np.where((((smag-testRadii-beamRadiusPhysical)<0 ) & (zmag>0)))
    

    
    testMass=testMass[mask]
    testSpeciesMassFrac=testSpeciesMassFrac[mask]
    testVel=testVel[mask]
    testRadii = testRadii[mask]
    testTemp = testTemp[mask]
    zmag = zmag[mask]
    smag = smag[mask]
    del mask
    

    dopplerVelocity = np.dot(testVel,zHat)
    dopplerVelocity = np.add(dopplerVelocity , (zmag-rObserver) * 0.07) #kpc * km/s /kpc, Hubble flow but still centered on galaxy



    #mask = np.where((((smag-testRadii-beamRadiusPhysical)<0 ) & (zmag>0)))
    indices = np.argsort(zmag) #start with the closest particles
    
    #Calculate the spectrum and emission from these particles
    spectrum,emission,optical_depth,nu = GenerateSpectra(testMass[indices],testSpeciesMassFrac[indices],dopplerVelocity[indices],testRadii[indices],testTemp[indices],zmag[indices],smag[indices],species=speciesToRun,beamSize=beamSize,Nspec=Nspec,bandwidth=bandwidth,calcThermalLevels=calcThermalLevels)
 

    return spectrum,emission
    



