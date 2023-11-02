import numpy as np
import h5py as h5py
import os.path
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import math
import sys
import time
import scipy.stats as stats

from VeryObservableFIRE.VOF_BinData import BinData3d
from VeryObservableFIRE.VOF_BinData import FlattenBinnedData
from VeryObservableFIRE.VOF_FindRotationCurve import FindRotationCurve
from VeryObservableFIRE.VOF_LoadData import ReadStats
from VeryObservableFIRE.VOF_LoadData import ReadSightlineFile
from VeryObservableFIRE.VOF_LoadData import LoadDataForSightlineIteration
from VeryObservableFIRE.VOF_LoadData import LoadSpeciesMassFrac
from VeryObservableFIRE.VOF_OrientGalaxy import CenterOnObserver
from VeryObservableFIRE.VOF_OrientGalaxy import OrientGalaxy
from VeryObservableFIRE.VOF_OrientGalaxy import GetRadialVelocity
from VeryObservableFIRE.VOF_LoadBinfireDataCube import LoadBinfireDataCube
from VeryObservableFIRE.VOF_GetParticlesInSightline import GetParticlesInSightline
from VeryObservableFIRE.VOF_GenerateSpectra import GenerateSpectra
from VeryObservableFIRE.VOF_EmissionSpecies import GetEmissionSpeciesParameters

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



        

def MakeMassImage(snapdir,Nsnapstring,statsDir,Nsightlines1d,output,sightlineDirBase,speciesToRun):
    pos_center,vel_center,Lhat,r0,orientation_maxima = ReadStats(statsDir);
    totalMass = np.zeros((Nsightlines1d,Nsightlines1d))
    for i in range(0,Nsightlines1d):
      print("Making Mass Image...",i,"/",Nsightlines1d)
      for j in range(0,Nsightlines1d):
        n = i*Nsightlines1d+j
        sightlineDir = sightlineDirBase+str(n)+".hdf5"
        mask,impact,distance,pos_observer,vel_observer,sightline = ReadSightlineFile(sightlineDir,tid)
        if (np.size(impact)>0):
          gPos,gVel,gMass,gKernal,gTemp =  LoadDataForSightlineIteration(snapdir,Nsnapstring,0,mask,pos_center,vel_center); #Load positions, generate truncmask, load the rest.
        
          for species in speciesToRun:
            speciesMassFrac = LoadSpeciesMassFrac(snapdir,Nsnapstring,0,mask,species)#,Gmas=gMass,KernalLengths=gKernal) #May be necesarry to load entire metallicity entry? CHECK!!!!
        
          totalMass[i,j] = np.sum(np.multiply(speciesMassFrac,gMass)) / np.sum(gMass)
          #totalMass[i,j] = np.mean(speciesMassFrac)



    #plt.imshow(totalMass,norm=LogNorm(),cmap='inferno')
    plt.imshow(totalMass,cmap='inferno')
    plt.colorbar()
    plt.savefig(output)
        



###NEED TO DO LoadDataForSightlineIteration (Specifically readsnap_trunc_sightline_itr and only loading 1 metallicity field at a time), and LoadSpeciesMassFrac (same issue)
def IterateSightlines(tid,snapdir,Nsnapstring,statsDir,output,sightlineDir,speciesToRun,beamSize,Nspec,bandwidths,calcThermalLevels=False):
    pos_center,vel_center,Lhat,r0,orientation_maxima = ReadStats(statsDir);

    mask,impact,distance,pos_observer,vel_observer,sightline = ReadSightlineFile(sightlineDir,tid)

    gPos,gVel,gMass,gKernal,gTemp =  LoadDataForSightlineIteration(snapdir,Nsnapstring,0,mask,pos_center,vel_center); #Load positions, generate truncmask, load the rest.
    gPos,gVel = OrientGalaxy(gPos,gVel,Lhat,r0);
    

    #gPos -= pos_observer
    #gVel -= vel_observer

    #rHat = np.copy(gPos)
    #rMag = VectorArrayMag(gPos)
   # rMag[rMag==0]=eps
   # rHat[:,0] = np.divide(gPos[:,0] , rMag)
   # rHat[:,1] = np.divide(gPos[:,1] , rMag)
   # rHat[:,2] = np.divide(gPos[:,2] , rMag)

    #rVelMag = np.multiply(gVel[:,0],rHat[:,0]) + np.multiply(gVel[:,1],rHat[:,1]) + np.multiply(gVel[:,2],rHat[:,2])
    
   # rVel = np.copy(gVel)
    #rVel[:,0] = np.multiply(rHat[:,0],rVelMag[:])
    #rVel[:,1] = np.multiply(rHat[:,1],rVelMag[:])
   # rVel[:,2] = np.multiply(rHat[:,2],rVelMag[:])
    
    #rVelProjectedMag = (np.multiply(rVel[:,0],sightline[0]) + np.multiply(rVel[:,1],sightline[1]) + np.multiply(rVel[:,2],sightline[2])) / np.linalg.norm(sightline)

    #hf = h5py.File(sightlineDir,'r+')

   # hf.create_dataset("total_radial_mass_flux",data=np.sum( np.divide( np.multiply( rVelMag, gMass) , rMag))) #Simple observational estimate for mass flux. Can be more thorough a la putman et al. 2012
    #accMask = np.where(rVelMag<0)
    #hf.create_dataset("total_radial_mass_flux_accreted_only",data=np.sum( np.divide( np.multiply( rVelMag[accMask], gMass[accMask]) , rMag[accMask])))
    #accMask = np.where(rVelProjectedMag<0)
    #hf.create_dataset("observed_radial_mass_flux",data=np.sum( np.divide( np.multiply( rVelProjectedMag, gMass) , rMag)))
    #hf.create_dataset("observed_radial_mass_flux_accreted_only",data=np.sum( np.divide( np.multiply( rVelProjectedMag[accMask], gMass[accMask]) , rMag[accMask])))

    
   # del rVelMag
   # del rVelProjectedMag
   # del rVel
   # del rHat
    
    

    
    #gPos_Sph,gRvel,gRhat = ConvertToSpherical(gPos,gVel,observer_position[idx]); #replace Gpos, Spos
    gDoppler = GetRadialVelocity(gPos-pos_observer,gVel-vel_observer); #replace Gpos, Spos
    i=0
    
    #print ("IN SIGHTLINE ITR, SUM OF MASS IS: ",np.sum(gMass));
    
    for species in speciesToRun:
        speciesMassFrac = LoadSpeciesMassFrac(snapdir,Nsnapstring,0,mask,species,Gmas=gMass,KernalLengths=gKernal) #May be necesarry to load entire metallicity entry? CHECK!!!!
        spectrum,emission,optical_depth,nu = GenerateSpectra(gMass,speciesMassFrac,gDoppler,gKernal,gTemp,distance,impact,species=species,beamSize=beamSize,Nspec=Nspec,bandwidth=bandwidths[i],calcThermalLevels=calcThermalLevels)
        #return spectrum
        return emission
       # WriteSpectra(spectrum,emission,optical_depth,nu,species,hf)
       # i+=1
    #hf.create_dataset("total_mass",data=np.sum(gMass))

   # hf.close()

     
def WriteSpectra(spectrum,emission,optical_depth,nu,species,hf):
  hf.create_dataset(species+"_spectrum",data=spectrum)
  hf.create_dataset(species+"_emission",data=emission)
  hf.create_dataset(species+"_optical_depth",data=optical_depth)
  hf.create_dataset(species+"_frequencies",data=nu)





def VectorArrayMag(r):
    r_magnitude = np.sqrt(np.add(np.power(r[:,0],2),np.add(np.power(r[:,1],2),np.power(r[:,2],2))))
    return r_magnitude







