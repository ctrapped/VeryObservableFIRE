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


def CenterOnObserver(observer_position,observer_velocity=None,rotationCurve=None,max_r=None):
    #Observer position and velocity in (pseudo)cylindrical coordinates?
    r, phi, z = observer_position
    pos_observer = [r * np.cos(phi) , r * np.sin(phi), z ]

    #Define Observer Velocity from RC if not provided
    if (observer_velocity is None):
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

def PredictVelocityFromRotationCurve(rmag,phi,rotationCurve,max_r):

    Npart = np.shape(rmag)[0]
    print("Npart is:",Npart)
    rInterp = np.linspace(0,max_r,np.size(rotationCurve),endpoint=True)
    f = interpolate.interp1d(rInterp , rotationCurve)
    vphi=np.zeros((Npart))
    for i in range(0,Npart):
        try:
            vphi[i] =  f(rmag[i]) #does this support just shoving an array in? Test??
        except:
            print("Warning: rotation curve not defined to high enough radii for these criteria, extrapolating")
            vphi[i] = (f(max_r) - f(max_r - 1)) * (rmag[i]-max_r)

    vel_predicted = np.zeros((Npart,3))
    vel_predicted[:,0] =  -np.multiply(vphi , np.sin(phi))
    vel_predicted[:,1] =  np.multiply(vphi , np.cos(phi))
    vel_predicted[:,2] = 0

    return vel_predicted
   
def OrientGalaxy(pos,vel,Lhat,r0,returnRotationalVelocity=False):
    N,dim = np.shape(pos)
    r_z = np.zeros((N,3))
    r_s = np.zeros((N,3))

    smag = np.zeros((N))
    zmag = np.zeros((N))

    zmag = np.dot(pos,Lhat)
    r_z[:,0] = zmag*Lhat[0]
    r_z[:,1] = zmag*Lhat[1]
    r_z[:,2] = zmag*Lhat[2]

    r_s = np.subtract(pos,r_z)
    smag = VectorArrayMag(r_s)
    smag[smag==0] = eps #make zero entries epsilon for division purposes

    s_hat = np.zeros((N,3))
    t_hat = np.zeros((N,3))

    s_hat[:,0] = np.divide(r_s[:,0],smag)
    s_hat[:,1] = np.divide(r_s[:,1],smag)
    s_hat[:,2] = np.divide(r_s[:,2],smag)
    t_hat = np.cross(Lhat,s_hat)


   # phi = np.zeros((N))

   # acos_term = np.divide(np.dot(r_s,r0),(np.linalg.norm(r0)*smag))
   # acos_term[acos_term>1] = 1 #make sure the term isn't above magnitude 1 by a rounding error
   # acos_term[acos_term<-1] = -1
   # phi = np.multiply( np.arccos(acos_term) , np.sign(np.dot(np.cross(r0,r_s),Lhat))) #first term gets us |phi| from 0 to pi, second term gives us the sign
    #del acos_term

   # phi[phi<0] = phi[phi<0] + 2*pi #make all values range from 0 to pi

    #xmag = np.multiply(smag,np.cos(phi))
    #ymag = np.multiply(smag,np.sin(phi))

    xhat = r0
    zhat = Lhat
    yhat = np.cross(zhat,xhat)
    
    pos_tmp = np.copy(pos)
    pos[:,0] = np.multiply(pos_tmp[:,0],xhat[0]) + np.multiply(pos_tmp[:,1],xhat[1]) + np.multiply(pos_tmp[:,2],xhat[2]);
    pos[:,1] = np.multiply(pos_tmp[:,0],yhat[0]) + np.multiply(pos_tmp[:,1],yhat[1]) + np.multiply(pos_tmp[:,2],yhat[2]);
    pos[:,2] = np.multiply(pos_tmp[:,0],zhat[0]) + np.multiply(pos_tmp[:,1],zhat[1]) + np.multiply(pos_tmp[:,2],zhat[2]);
    #pos[:,0] = xmag
    #pos[:,1] = ymag
    #pos[:,2] = zmag

    if vel is not None:
        vel_tmp = np.copy(vel)
        vel[:,0] = np.multiply(vel_tmp[:,0],xhat[0]) + np.multiply(vel_tmp[:,1],xhat[1]) + np.multiply(vel_tmp[:,2],xhat[2]);
        vel[:,1] = np.multiply(vel_tmp[:,0],yhat[0]) + np.multiply(vel_tmp[:,1],yhat[1]) + np.multiply(vel_tmp[:,2],yhat[2]);
        vel[:,2] = np.multiply(vel_tmp[:,0],zhat[0]) + np.multiply(vel_tmp[:,1],zhat[1]) + np.multiply(vel_tmp[:,2],zhat[2]);

        if returnRotationalVelocity:
            #rotVel = np.multiply(vel_tmp[:,0],t_hat[:,0]) + np.multiply(vel_tmp[:,1],t_hat[:,1]) + np.multiply(vel_tmp[:,2],t_hat[:,2]);
            j = np.cross(pos_tmp, vel_tmp)
            jz = np.multiply(j[:,0],zhat[0]) + np.multiply(j[:,1],zhat[1]) + np.multiply(j[:,2],zhat[2])
            rmag = VectorArrayMag(pos_tmp)
            rotVel = np.divide(jz , rmag)
            #rotVel = np.cross(vel_tmp,s_hat) + np.cross(vel_tmp,t_hat)
            return pos,vel,rotVel#[:,2]

        return pos,vel
    else:
        return pos
 
def GetRadialVelocity(pos,vel):
    N,dim = np.shape(pos)

    rMag = VectorArrayMag(pos)
    rHat = np.copy(pos)
    rHat[:,0] = np.divide(pos[:,0],rMag)
    rHat[:,1] = np.divide(pos[:,1],rMag)
    rHat[:,2] = np.divide(pos[:,2],rMag)

    rVel = np.multiply(vel[:,0],rHat[:,0]) + np.multiply(vel[:,1],rHat[:,1]) + np.multiply(vel[:,2],rHat[:,2])
    return rVel

def ConvertToSpherical(pos,vel,observer_position):
    pos_galCenter = -np.array([observer_position[0] * np.cos(observer_position[1]) , observer_position[0] * np.sin(observer_position[1]), observer_position[2] ])
    N,dim = np.shape(pos)

    rMag = VectorArrayMag(pos)
    rHat = np.copy(pos)
    rHat[:,0] = np.divide(pos[:,0],rMag)
    rHat[:,1] = np.divide(pos[:,1],rMag)
    rHat[:,2] = np.divide(pos[:,2],rMag)
    #del pos

    
    #rVel = np.zeros((N))
    rVel = np.multiply(vel[:,0],rHat[:,0]) + np.multiply(vel[:,1],rHat[:,1]) + np.multiply(vel[:,2],rHat[:,2])
    #rVel = np.dot(vel,np.transpose(rHat))

    sMag = np.sqrt(np.add(np.power(pos[:,0],2),np.power(pos[:,1],2)))
    pos[pos[:,2]==0,2]=eps #avoid divide by 0
    theta = np.arctan(np.divide(sMag,pos[:,2]))
    theta[theta>0] = pi/2. - theta[theta>0]
    theta[theta<0] = -pi/2 - theta[theta<0]

    #Set Phi=0 at galactic center
    r_s = np.zeros((N,3))
    r_s[:,0] = pos[:,0]
    r_s[:,1] = pos[:,1]
    r0 = pos_galCenter
    Lhat = [0,0,1]
    acos_term = np.divide(np.dot(r_s,r0),(np.linalg.norm(r0)*sMag))
    acos_term[acos_term>1] = 1 #make sure the term isn't above magnitude 1 by a rounding error
    acos_term[acos_term<-1] = -1
    phi = np.multiply( np.arccos(acos_term) , np.sign(np.dot(np.cross(r0,r_s),Lhat))) #first term gets us |phi| from 0 to pi, second term gives us the sign
    #phi[phi<0] = phi[phi<0] + 2*pi #make all values range from 0 to pi
    
    pos[:,0] = rMag
    pos[:,1] = theta
    pos[:,2] = phi
    return pos,rVel,rHat


def VectorArrayMag(r):
    r_magnitude = np.sqrt(np.add(np.power(r[:,0],2),np.add(np.power(r[:,1],2),np.power(r[:,2],2))))
    return r_magnitude









