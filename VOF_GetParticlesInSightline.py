import numpy as np
import math
import sys
import time

unit_M = 10**10 * 1.98855 *10**33 #10^10 solar masses / h !!in grams!! #h accounted for in readsnap
unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
unit_V = 1.0*10.0**5 # 1 km/s !!in cm/s!! Converted by factor of sqrt(a) in readsnap
unit_T = unit_L/unit_V
unit_rho = unit_M / unit_L**3

proton_mass = 1.6726219*10**(-27)*(1000.0/unit_M) ##appropriate units

pi = np.pi
eps = 0.00000000000000000000000000000000000000000000001

def GetParticlesInSightline(sightline,testPositions,testRadii,pids,cids,gen,beamSize,rmag,max_r): #assume already centered on observer
    print("Setting up sightline geometry...")
    radMask = np.where(rmag<max_r*1.5)[0]
    zHat = sightline / np.linalg.norm(sightline) #orient cylindrical coordinate system along the sightline
    N,dim = np.shape(testPositions)
    r_z = np.zeros((N,3)) #vector for the z component in the old coord system

    zmag = np.zeros((N)) 
    zmag[radMask] = np.dot(testPositions[radMask,:],zHat)
    r_z[radMask,0] = zmag[radMask]*zHat[0]
    r_z[radMask,1] = zmag[radMask]*zHat[1]
    r_z[radMask,2] = zmag[radMask]*zHat[2]

    r_s=np.zeros((N,3))
    r_s[radMask,:] = np.subtract(testPositions[radMask,:],r_z[radMask,:]) #vector for the s component in the old coord system
    del r_z
    smag=np.zeros((N))
    smag[radMask] = VectorArrayMag(r_s[radMask,:])
    
    beamRadiusPhysical=np.zeros((N))
    #beamRadiusPhysical = zmag * np.sin(beamSize)/2 #the physical size of the beam for each particle
    beamRadiusPhysical[radMask] = zmag[radMask] * beamSize / 2 #the physical size of the beam for each particle
    #print("beamRadius Physical range from ",np.min(beamRadiusPhysical),'-',np.max(beamRadiusPhysical),' kpc')
    print("Defining mask...")
    mask = np.where((((smag-testRadii-beamRadiusPhysical)<0 ) & (zmag>0) & (rmag<max_r*1.5)))
    print("Sorting...")
    indices = np.argsort(zmag[mask]) #start with the closest particles

    return mask[0][indices],smag[mask][indices],zmag[mask][indices],pids[mask][indices],cids[mask][indices],gen[mask][indices] #smag is impact parameter
    
    
def GetHVCParticlesInSightline(sightline,testPositions,testRadii,testVelocities,rcPredictedVelocities,pids,cids,gen,beamSize,dv_min=None,dv_max=None): #assume already centered on observer

    zHat = sightline / np.linalg.norm(sightline) #orient cylindrical coordinate system along the sightline
    N,dim = np.shape(testPositions)
    r_z = np.zeros((N,3)) #vector for the z component in the old coord system

    zmag = np.zeros((N)) 
    zmag = np.dot(testPositions,zHat)
    r_z[:,0] = zmag*zHat[0]
    r_z[:,1] = zmag*zHat[1]
    r_z[:,2] = zmag*zHat[2]
    
    
    dopperVelocity = np.dot(testVelocities,zHat);
    deltaVelocity = np.dot(rcPredictedVelocities,zHat) - dopperVelocity;
    r_s = np.subtract(testPositions,r_z) #vector for the s component in the old coord system
    del r_z
    smag = VectorArrayMag(r_s)

    beamRadiusPhysical = zmag * np.sin(beamSize)/2 #the physical size of the beam for each particle

    if ((dv_min is not None) and (dv_max is not None)):
        mask = np.where((((smag-testRadii-beamRadiusPhysical)<0 ) & (zmag>0) & (np.abs(deltaVelocity) > dv_min) & (np.abs(deltaVelocity) < dv_max)))
    elif (dv_min is not None):
        mask = np.where((((smag-testRadii-beamRadiusPhysical)<0 ) & (zmag>0) & (np.abs(deltaVelocity) > dv_min)))
    elif (dv_max is not None):
        mask = np.where((((smag-testRadii-beamRadiusPhysical)<0 ) & (zmag>0) & (np.abs(deltaVelocity) < dv_max)))
    else:
        mask = np.where((((smag-testRadii-beamRadiusPhysical)<0 ) & (zmag>0)))
    indices = np.argsort(zmag[mask]) #start with the closest particles

    return mask[0][indices],smag[mask][indices],zmag[mask][indices],pids[mask][indices],cids[mask][indices],gen[mask][indices] #smag is impact parameter

    
def RC2DopplerVelocity(rotationCurve , rmag, max_r):

    rInterp = np.linspace(0,max_r,np.size(rotationCurve),endpoint=True)
    f = interpolate.interp1d(rInterp , rotationCurve)
    observer_velocity = [0, f(rmag), 0]

    vel_observer = np.zeros((3))
    vr,vphi,vz = observer_velocity
    vel_observer[0] = vr * math.cos(phi) - vphi * math.sin(phi)
    vel_observer[1] = vr * math.sin(phi) + vphi * math.cos(phi)
    vel_observer[2] = vz

def VectorArrayMag(r):
    r_magnitude = np.sqrt(np.add(np.power(r[:,0],2),np.add(np.power(r[:,1],2),np.power(r[:,2],2))))
    return r_magnitude




