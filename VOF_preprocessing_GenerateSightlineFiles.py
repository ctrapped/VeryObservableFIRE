import numpy as np
import h5py as h5py
import os.path
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import math
import sys
import time
import scipy.stats as stats

import scipy

from VeryObservableFIRE.VOF_LoadData import ReadStats
from VeryObservableFIRE.VOF_LoadData import LoadDataForSightlineGenerator
from VeryObservableFIRE.VOF_LoadData import LoadDataForSightlineIteration

from VeryObservableFIRE.VOF_LoadData import LoadDataForHVCSightlineGenerator
from VeryObservableFIRE.VOF_LoadData import LoadData
from VeryObservableFIRE.VOF_FindRotationCurve import FindRotationCurve
from VeryObservableFIRE.VOF_OrientGalaxy import CenterOnObserver
from VeryObservableFIRE.VOF_OrientGalaxy import OrientGalaxy
from VeryObservableFIRE.VOF_OrientGalaxy import PredictVelocityFromRotationCurve
from VeryObservableFIRE.VOF_GetParticlesInSightline import GetParticlesInSightline
from VeryObservableFIRE.VOF_GetParticlesInSightline import GetHVCParticlesInSightline
from VeryObservableFIRE.VOF_LoadData import LoadSpeciesMassFrac

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

unit_M = 10**10 * 1.98855 *10**33 #10^10 solar masses / h !!in grams!! #h accounted for in readsnap
unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
unit_V = 1.0*10.0**5 # 1 km/s !!in cm/s!! Converted by factor of sqrt(a) in readsnap
unit_T = unit_L/unit_V
unit_rho = unit_M / unit_L**3

proton_mass = 1.6726219*10**(-27)*(1000.0/unit_M) ##appropriate units

pi = np.pi
eps = 0.00000000000000000000000000000000000000000000001

arcsec2rad = pi / (180*3600)


def GenerateSightlineFiles(snapdir,Nsnapstring,statsDir,observer_position,observer_velocity,outputs,maxima,beamSize = 1*arcsec2rad,Nsightlines=100,sightlines=None,phiObs=0,inclination=0):
    
    max_r,maxPhi,maxTheta = maxima;

    pos_center,vel_center,Lhat,r0,orientation_maxima = ReadStats(statsDir);
    
    if inclination>0:
        rotation_axis = np.cross(Lhat,r0)
        rotation_vector = float(inclination)*np.pi/180.*rotation_axis
        rotation = scipy.spatial.transform.Rotation.from_rotvec(rotation_vector)
        print("IN GENERATE SIGHTLINES FILE INCLINATION=", float(inclination)*np.pi/180.," rads")

        Lhat = rotation.apply(Lhat)
        r0 = rotation.apply(r0)
    else:
        print("WARNING!! 0 INCLINATION!!!")


    gPos,gKernal,gPids,gCids,gGen = LoadDataForSightlineGenerator(snapdir,Nsnapstring,0,max_r,pos_center,vel_center); #Load positions, generate truncmask, load the rest.
    gPos,phi = OrientGalaxy(gPos,None,Lhat,r0,True);
    
    rmag = VectorArrayMag(gPos)

    sPos,sVel,sMass = LoadData(snapdir,Nsnapstring,1,max_r,pos_center,vel_center);
    sPos,sVel,phi = OrientGalaxy(sPos,sVel,Lhat,r0,True);

    rBinSize=0.1#Fix up!!
    nr = int(math.ceil(max_r/rBinSize)) 
    rotationCurve = FindRotationCurve(sPos,sVel,sMass,nr,max_r); #Make optional


    Nobservers = 1
    for idx in range(0,Nobservers):
      output = outputs
      pos_observer,vel_observer = CenterOnObserver(observer_position[idx],observer_velocity[idx],rotationCurve=rotationCurve,max_r=max_r);

      gPos -= pos_observer
      #PARALLELIZE HERE, Write masks for particles in LOS separately, load only the particles of interest here
      #sightlines = [np.array(pos_observer)] #just look towards center right now
      #beamSize = 1 * arcsec2rad #Make an input argument

      if sightlines is None:
        Nsightlines_1d = int(np.round(np.sqrt(Nsightlines)))
        sightlines = np.zeros((Nsightlines_1d,Nsightlines_1d,3))
        phiRes = maxPhi / (Nsightlines_1d-1)
        thetaRes = maxTheta / (Nsightlines_1d-1)
        
        phi0 = (2*pi - maxPhi)/2 + phiObs - pi
        theta0 = (pi - maxTheta)/2
        
        #Issue with symmetry of theta? set thetares to maxTheta/(Nsightlines-1)??
        
        indices = np.indices((Nsightlines_1d,Nsightlines_1d))
        sightlines[:,:,0] = np.cos(indices[0,:,:]*phiRes+phi0)*np.cos(-(indices[1,:,:]*thetaRes-pi/2+theta0)) #May need to flip a sign or something? Check on run!
        sightlines[:,:,1] = np.sin(indices[0,:,:]*phiRes+phi0)*np.cos(-(indices[1,:,:]*thetaRes-pi/2+theta0))
        sightlines[:,:,2] = np.sin(-(indices[1,:,:]*thetaRes-pi/2+theta0))

      print("sightlines=", '[',np.min(sightlines[:,:,0]),',',np.max(sightlines[:,:,0]),']')
      print( '[',np.min(sightlines[:,:,1]),',',np.max(sightlines[:,:,1]),']')
      print( '[',np.min(sightlines[:,:,2]),',',np.max(sightlines[:,:,2]),']')
      

      #MakeInitialMassImage(gPos,pos_observer,snapdir,Nsnapstring,output+"_HI_mass.png",pos_center,vel_center)
      
      for i in range(0,Nsightlines_1d):
        for j in range(0,Nsightlines_1d):
          print("Generating sightline [",i,',',j,']')
          mask,impact,distance,pids,cids,gen=GetParticlesInSightline(sightlines[i,j,:],gPos,gKernal,gPids,gCids,gGen,beamSize,rmag,max_r)
          saveFile = output+"_sightline"+str(i*Nsightlines_1d+j)+".hdf5"
          SaveSightline(sightlines[i,j,:],mask,impact,distance,pids,cids,gen,pos_observer,vel_observer,saveFile)
      gPos += pos_observer
      
      

def MakeInitialMassImage(gPos,pos_observer,snapdir,Nsnapstring,output,pos_center,vel_center):
    gPos[:,0]=gPos[:,0]+pos_observer[0]
    gPos[:,1]=gPos[:,1]+pos_observer[1]
    gPos[:,2]=gPos[:,2]+pos_observer[2]

    mask = np.where(np.abs(gPos[:,0]>=0))[0]
    tmp,gVel,gMass,gKernal,gTemp =  LoadDataForSightlineIteration(snapdir,Nsnapstring,0,None,pos_center,vel_center); #Load positions, generate truncmask, load the rest.

    for species in ['HI_21cm']:
        speciesMassFrac = LoadSpeciesMassFrac(snapdir,Nsnapstring,0,None,species)#,Gmas=gMass,KernalLengths=gKernal) #May be necesarry to load entire metallicity entry? CHECK!!!!
    hiMass = np.multiply(gMass,speciesMassFrac)
    
    
    max_x=20
    max_y=20
    nx=40
    ny=40
    sample = np.zeros((np.size(hiMass),2))
    print(np.size(mask))
    print(np.shape(gPos),np.shape(gMass),np.shape(speciesMassFrac),np.shape(hiMass),np.shape(sample))

    sample[:,0] = gPos[:,1]
    sample[:,1] = gPos[:,2]
    
    binnedHi,tmp,tmp = stats.binned_statistic_dd(sample,hiMass,'sum',[nx,ny],range=[[-max_x,max_x],[-max_y,max_y]])
    plt.imshow(binnedHi,cmap='inferno')
    plt.savefig(output)


def GenerateHVCSightlineFiles(snapdir,Nsnapstring,statsDir,observer_position,observer_velocity,outputs,maxima,beamSize = 1*arcsec2rad,Nsightlines=100,sightlines=None):
    
    max_r,max_theta,max_phi = maxima;

    pos_center,vel_center,Lhat,r0,orientation_maxima = ReadStats(statsDir);

    gPos,gVel,gMass,gKernal,gPids,gCids,gGen = LoadDataForHVCSightlineGenerator(snapdir,Nsnapstring,0,max_r,pos_center,vel_center); #Load positions, generate truncmask, load the rest.
    gPos,gVel,gPhi = OrientGalaxy(gPos,gVel,Lhat,r0,True);

    sPos,sVel,sMass = LoadData(snapdir,Nsnapstring,1,max_r,pos_center,vel_center);
    sPos,sVel,sPhi = OrientGalaxy(sPos,sVel,Lhat,r0,True);

    rBinSize=0.1#Fix up!!
    nr = int(math.ceil(max_r/rBinSize)) 
    rotationCurve = FindRotationCurve(sPos,sVel,sMass,nr,max_r); #Make optional
    rotationCurveGas = FindRotationCurve(gPos,gVel,gMass,nr,max_r);
    gVel_RC_Prediction = PredictVelocityFromRotationCurve(VectorArrayMag(gPos),gPhi,rotationCurveGas,max_r);


    Nobservers = 1
    for idx in range(0,Nobservers):
      output = outputs
      pos_observer,vel_observer = CenterOnObserver(observer_position[idx],observer_velocity[idx],rotationCurve=rotationCurve,max_r=max_r);

      gPos -= pos_observer
      gVel -= vel_observer
      gVel_RC_Prediction -= vel_observer
      
      #PARALLELIZE HERE, Write masks for particles in LOS separately, load only the particles of interest here
      #sightlines = [np.array(pos_observer)] #just look towards center right now
      #beamSize = 1 * arcsec2rad #Make an input argument

      if sightlines is None:
        Nsightlines_1d = int(np.round(np.sqrt(Nsightlines)))
        sightlines = np.zeros((Nsightlines_1d,Nsightlines_1d,3))
        phiRes = 2*pi / Nsightlines_1d
        thetaRes = pi / Nsightlines_1d
        indices = np.indices((Nsightlines_1d,Nsightlines_1d))
        sightlines[:,:,0] = np.cos(indices[0,:,:]*phiRes)*np.cos(indices[1,:,:]*thetaRes-pi/2) #May need to flip a sign or something? Check on run!!!!!!!!!!!!!
        sightlines[:,:,1] = np.sin(indices[0,:,:]*phiRes)*np.cos(indices[1,:,:]*thetaRes-pi/2)
        sightlines[:,:,2] = np.sin(indices[1,:,:]*thetaRes-pi/2)

      dv_min = 70 #HVC cutoff
      dv_max= None
      for i in range(0,Nsightlines_1d):
        for j in range(0,Nsightlines_1d):
          mask,impact,distance,pids,cids,gen=GetHVCParticlesInSightline(sightlines[i,j,:],gPos,gKernal,gVel,gVel_RC_Prediction,gPids,gCids,gGen,beamSize,dv_min,dv_max)
          #saveFile = output+"_sightline"+str(i*Nsightlines_1d+j)+".hdf5"
          saveFile = output+"_allSightlines.hdf5"
          Nsightline = i*Nsightlines_1d+j
          SaveSightline(sightlines[i,j,:],mask,impact,distance,pids,cids,gen,pos_observer,vel_observer,saveFile,Nsightline)
      gPos += pos_observer



def SaveSightline(sightline,mask,impact,distance,pids,cids,gen,pos_observer,vel_observer,saveFile,Nsightline):
 # hf = h5py.File(saveFile,'w')
  hf = h5py.File(saveFile,'r+')
  groupName = "sightline"+str(Nsightline)
  hf.create_group(groupName)
  hf[groupName].create_dataset("sightline",data=sightline)
  hf[groupName].create_dataset("mask",data=mask)
  hf[groupName].create_dataset("impact",data=impact)
  hf[groupName].create_dataset("pids",data=pids)
  hf[groupName].create_dataset("cids",data=cids)
  hf[groupName].create_dataset("gen",data=gen)
  hf[groupName].create_dataset("distance",data=distance)
  hf[groupName].create_dataset("pos_observer",data=pos_observer)
  hf[groupName].create_dataset("vel_observer",data=vel_observer)
  hf.close()



def VectorArrayMag(r):
    r_magnitude = np.sqrt(np.add(np.power(r[:,0],2),np.add(np.power(r[:,1],2),np.power(r[:,2],2))))
    return r_magnitude







