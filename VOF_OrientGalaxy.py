import numpy as np
import h5py as h5py
import os.path
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import math
import sys
import time
import scipy.stats as stats


unit_M = 10**10 * 1.98855 *10**33 #10^10 solar masses / h !!in grams!! #h accounted for in readsnap
unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
unit_V = 1.0*10.0**5 # 1 km/s !!in cm/s!! Converted by factor of sqrt(a) in readsnap
unit_T = unit_L/unit_V
unit_rho = unit_M / unit_L**3

proton_mass = 1.6726219*10**(-27)*(1000.0/unit_M) ##appropriate units

pi = np.pi
eps = 0.00000000000000000000000000000000000000000000001


####Functions to center on observer and orient coordinate systems to certain vectors
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023

def CenterOnObserver(observer_position,observer_velocity=None,rotationCurve=None,max_r=None):
    #Observer position and velocity in (pseudo)cylindrical coordinates?
    r, phi, z = observer_position
    pos_observer = [r * np.cos(phi) , r * np.sin(phi), z ]

    #Define Observer Velocity from RC if not provided
    if observer_velocity is None:
        #Assume only moving azimuthally
        rInterp = np.linspace(0,max_r,np.size(rotationCurve),endpoint=True)
        f = interpolate.interp1d(rInterp , rotationCurve)
        observer_velocity = [0, f(r), 0]

    vel_observer = np.zeros((3))
    vr,vphi,vz = observer_velocity
    vel_observer[0] = vr * math.cos(phi) - vphi * math.sin(phi)
    vel_observer[1] = vr * math.sin(phi) + vphi * math.cos(phi)
    vel_observer[2] = vz

    return pos_observer , vel_observer

   
def OrientGalaxy(pos,vel,Lhat,r0,returnRotationalVelocity=False):
    #Rotate galaxy with Lhat defined as the z direction and r0 as the x direction
    xhat = r0
    zhat = Lhat
    yhat = np.cross(zhat,xhat)
    
    pos_tmp = np.copy(pos)
    pos[:,0] = np.multiply(pos_tmp[:,0],xhat[0]) + np.multiply(pos_tmp[:,1],xhat[1]) + np.multiply(pos_tmp[:,2],xhat[2]);
    pos[:,1] = np.multiply(pos_tmp[:,0],yhat[0]) + np.multiply(pos_tmp[:,1],yhat[1]) + np.multiply(pos_tmp[:,2],yhat[2]);
    pos[:,2] = np.multiply(pos_tmp[:,0],zhat[0]) + np.multiply(pos_tmp[:,1],zhat[1]) + np.multiply(pos_tmp[:,2],zhat[2]);

    if vel is not None:
        vel_tmp = np.copy(vel)
        vel[:,0] = np.multiply(vel_tmp[:,0],xhat[0]) + np.multiply(vel_tmp[:,1],xhat[1]) + np.multiply(vel_tmp[:,2],xhat[2]);
        vel[:,1] = np.multiply(vel_tmp[:,0],yhat[0]) + np.multiply(vel_tmp[:,1],yhat[1]) + np.multiply(vel_tmp[:,2],yhat[2]);
        vel[:,2] = np.multiply(vel_tmp[:,0],zhat[0]) + np.multiply(vel_tmp[:,1],zhat[1]) + np.multiply(vel_tmp[:,2],zhat[2]);

        if returnRotationalVelocity:
            j = np.cross(pos_tmp, vel_tmp)
            jz = np.multiply(j[:,0],zhat[0]) + np.multiply(j[:,1],zhat[1]) + np.multiply(j[:,2],zhat[2])
            rmag = VectorArrayMag(pos_tmp)
            rotVel = np.divide(jz , rmag)
            return pos,vel,rotVel

        return pos,vel
    else:
        return pos
 
def GetRadialVelocity(pos,vel,rObs):
    N,dim = np.shape(pos)

    rMag = VectorArrayMag(pos)
    rHat = np.copy(pos)
    rHat[:,0] = np.divide(pos[:,0],rMag)
    rHat[:,1] = np.divide(pos[:,1],rMag)
    rHat[:,2] = np.divide(pos[:,2],rMag)

    rVel = np.multiply(vel[:,0],rHat[:,0]) + np.multiply(vel[:,1],rHat[:,1]) + np.multiply(vel[:,2],rHat[:,2])
    
    rVel = np.add( rVel , (rMag-rObs) * 0.07) #kpc * km/s /kpc, Hubble flow but still centered on galaxy
    
    return rVel



def VectorArrayMag(r):
    r_magnitude = np.sqrt(np.add(np.power(r[:,0],2),np.add(np.power(r[:,1],2),np.power(r[:,2],2))))
    return r_magnitude









