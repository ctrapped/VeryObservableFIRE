import numpy as np
import h5py as h5py
import math
import time

import scipy

from VOF_LoadData import ReadStats
from VOF_LoadData import LoadDataForSightlineGenerator
from VOF_LoadData import LoadData

from VOF_FindRotationCurve import FindRotationCurve
from VOF_OrientGalaxy import CenterOnObserver
from VOF_OrientGalaxy import OrientGalaxy
from VOF_GetParticlesInSightline import GetParticlesInSightline

from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool

from functools import partial

unit_M = 10**10 * 1.98855 *10**33 #10^10 solar masses / h !!in grams!! #h accounted for in readsnap
unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
unit_V = 1.0*10.0**5 # 1 km/s !!in cm/s!! Converted by factor of sqrt(a) in readsnap
unit_T = unit_L/unit_V
unit_rho = unit_M / unit_L**3

proton_mass = 1.6726219*10**(-27)*(1000.0/unit_M) ##appropriate units

pi = np.pi

arcsec2rad = pi / (180*3600)

####Functions to generate a sightline file for a given set of observer parameters. Defines the vectors, overlapping particles in each sightline,
####and relevant parameters that allows each thread to get the effective column density of a particle along that sightline.
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023

def foo(thread_id,Nsightlines_1d,sightlines,gPos,gKernal,beamSize,rmag,max_r):
    ix=thread_id % Nsightlines_1d #Thread pixel coordinates
    iy = int(np.floor(thread_id / Nsightlines_1d))
    print("Generating sightline [",ix,',',iy,']')
    mask,impact,distance=GetParticlesInSightline(sightlines[ix,iy,:],gPos,gKernal,beamSize,rmag,max_r)
    #Nsightline = i*Nsightlines_1d+j
    #saveFile = output+"_allSightlines.hdf5"
    # SaveSightline(sightlines[i,j,:],mask,impact,distance,pos_observer,vel_observer,saveFile,Nsightline) #Save information for parallel spectra generation

    return mask,impact,distance


def GenerateSightlineFiles(snapdir,Nsnapstring,statsDir,observer_position,observer_velocity,outputs,maxima,beamSize = 1*arcsec2rad,Nsightlines=100,sightlines=None,phiObs=0,inclination=0):
    max_r,maxPhi,maxTheta = maxima;

    pos_center,vel_center,Lhat,r0,orientation_maxima = ReadStats(statsDir);
    
    if inclination>0:
        #Rotate the vectors Lhat and r0 that define the z and x unit vectors respectively. Effectively rotates the entire galaxy
        rotation_axis = np.cross(Lhat,r0)
        rotation_vector = float(inclination)*np.pi/180.*rotation_axis
        rotation = scipy.spatial.transform.Rotation.from_rotvec(rotation_vector)

        Lhat = rotation.apply(Lhat)
        r0 = rotation.apply(r0)

    #Load the gas particles
    gPos,gKernal = LoadDataForSightlineGenerator(snapdir,Nsnapstring,0,max_r,pos_center,vel_center)
    #Transform into previously defined coordinate system
    gPos = OrientGalaxy(gPos,None,Lhat,r0);
    
    rmag = VectorArrayMag(gPos)
    
    #If observer velocity is not defined, calculate rotation curve to put the observer in the galaxy. Should only be used for in galaxy observations.
    rotationCurve=None
    defineRotationCurve=False
    
    if observer_velocity is None: defineRotationCurve=True
    
    if defineRotationCurve:
        #Load the star particles
        sPos,sVel,sMass = LoadData(snapdir,Nsnapstring,1,max_r,pos_center,vel_center);
        #Transform into previously defined coordinate system
        sPos,sVel = OrientGalaxy(sPos,sVel,Lhat,r0);

        rBinSize=0.1
        nr = int(math.ceil(max_r/rBinSize)) 
        rotationCurve = FindRotationCurve(sPos,sVel,sMass,nr,max_r)


    output = outputs
    pos_observer,vel_observer = CenterOnObserver(observer_position,observer_velocity,rotationCurve=rotationCurve,max_r=max_r);

    gPos -= pos_observer #Switch to observers frame of reference

    if sightlines is None: #Create sightline vectors to evenly sample the observed space
        Nsightlines_1d = int(np.round(np.sqrt(Nsightlines)))
        sightlines = np.zeros((Nsightlines_1d,Nsightlines_1d,3))
        phiRes = maxPhi / (Nsightlines_1d-1)
        thetaRes = maxTheta / (Nsightlines_1d-1)
        
        phi0 = (2*pi - maxPhi)/2 + phiObs - pi
        theta0 = (pi - maxTheta)/2
                
        indices = np.indices((Nsightlines_1d,Nsightlines_1d))
        sightlines[:,:,0] = np.cos(indices[0,:,:]*phiRes+phi0)*np.cos(-(indices[1,:,:]*thetaRes-pi/2+theta0))
        sightlines[:,:,1] = np.sin(indices[0,:,:]*phiRes+phi0)*np.cos(-(indices[1,:,:]*thetaRes-pi/2+theta0))
        sightlines[:,:,2] = np.sin(-(indices[1,:,:]*thetaRes-pi/2+theta0))

    t0=time.time()
    #For each sightline get the particles that overlap with the beam and their offset from the beam. Assumes particles are spheres (they aren't, this can be improved)
    sightline_indices = range(0,Nsightlines)
    num_cores = multiprocessing.cpu_count()-1
    if num_cores>16: num_cores=16
    print("Working with ",num_cores," cores")
    
    
    #indices = np.argsort(zmag[mask]) #start with the closest particles

   # print(np.shape(mask[0][indices]))
    #return mask[0][indices],smag[mask][indices],zmag[mask][indices] #smag is impact parameter
    
    foo_ = partial(foo,Nsightlines_1d=Nsightlines_1d,sightlines=sightlines,gPos=gPos,gKernal=gKernal,beamSize=beamSize,rmag=rmag,max_r=max_r)
    x= Parallel(n_jobs=num_cores)(delayed(foo_)(i) for i in sightline_indices)
    #with Pool(processes=num_cores) as pool:
    #    x= pool.map(foo_ , sightline_indices)
    print("Time to run in parallel=",time.time()-t0)
    #masks=x[:,0]
    #impacts=x[:,1]
    #distances=x[:,2]
    #print("Shape of masks=",np.shape(masks))    
    saveFile = output+"_allSightlines.hdf5"
    hf = h5py.File(saveFile,'w') #overwrite if it exists already
    hf.create_dataset("pos_observer",data=pos_observer)
    hf.create_dataset("vel_observer",data=vel_observer)
    for i in range(0,Nsightlines_1d):
          for j in range(0,Nsightlines_1d):
            Nsightline = i*Nsightlines_1d+j
            mask = x[Nsightline][0]
            impact = x[Nsightline][1]
            distance= x[Nsightline][2]
            SaveSightline(hf,sightlines[i,j,:],mask,impact,distance,Nsightline) #Save information for parallel spectra generation
    hf.close()
    

def SaveSightline(hf,sightline,mask,impact,distance,Nsightline):
  #Save all sightlines in their own file, but with group name equivalent to the thread id that will access it
  #hf = h5py.File(saveFile,'w') #overwrite if it exists already
  groupName = "sightline"+str(Nsightline)
  hf.create_group(groupName)
  hf[groupName].create_dataset("sightline",data=sightline)
  hf[groupName].create_dataset("mask",data=mask)
  hf[groupName].create_dataset("impact",data=impact)
  hf[groupName].create_dataset("distance",data=distance)
  #hf[groupName].create_dataset("pos_observer",data=pos_observer)
  #hf[groupName].create_dataset("vel_observer",data=vel_observer)
 # hf.close()



def VectorArrayMag(r):
    r_magnitude = np.sqrt(np.add(np.power(r[:,0],2),np.add(np.power(r[:,1],2),np.power(r[:,2],2))))
    return r_magnitude

