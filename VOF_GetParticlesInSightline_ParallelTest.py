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

####Function to find all particles that intersect with a given sightline of finite beam size
####Returns a mask for intersecting particles, as well as various parameters sorted by distance from the observer
####Such that absorption can be built up sequentially
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023


def GetParticlesInSightline(sightline,testPositions,testRadii,beamSize,rmag,max_r): #assume already centered on observer
    #print("Setting up sightline geometry...")
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
    beamRadiusPhysical[radMask] = zmag[radMask] * beamSize / 2 #the physical size of the beam for each particle
    #print("Defining mask...")
    mask = (((smag-testRadii-beamRadiusPhysical)<0 ) & (zmag>0) & (rmag<max_r*1.5))
    #print("Sorting...")
 
    return mask,smag,zmag
    #indices = np.argsort(zmag[mask]) #start with the closest particles

   # print(np.shape(mask[0][indices]))
    #return mask[0][indices],smag[mask][indices],zmag[mask][indices] #smag is impact parameter
    
    

def VectorArrayMag(r):
    r_magnitude = np.sqrt(np.add(np.power(r[:,0],2),np.add(np.power(r[:,1],2),np.power(r[:,2],2))))
    return r_magnitude




