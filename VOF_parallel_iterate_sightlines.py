import numpy as np
from VOF_LoadData import ReadStats
from VOF_LoadData import ReadSightlineFile
from VOF_LoadData import LoadDataForSightlineIteration
from VOF_LoadData import LoadSpeciesMassFrac
from VOF_OrientGalaxy import OrientGalaxy
from VOF_OrientGalaxy import GetRadialVelocity
from VOF_GenerateSpectra import GenerateSpectra

unit_M = 10**10 * 1.98855 *10**33 #10^10 solar masses / h !!in grams!! #h accounted for in readsnap
unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
unit_V = 1.0*10.0**5 # 1 km/s !!in cm/s!! Converted by factor of sqrt(a) in readsnap
unit_T = unit_L/unit_V
unit_rho = unit_M / unit_L**3

proton_mass = 1.6726219*10**(-27)*(1000.0/unit_M) ##appropriate units

pi = np.pi
eps = 1e-10

arcsec2rad = pi / (180*3600)

####Function to generate emission spectra along a given sightline.
####Sightlines and their intersecting particles are determined by VOF_GenerateSightlineFiles.py
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023

def IterateSightlines(tid,snapdir,Nsnapstring,statsDir,output,sightlineDir,speciesToRun,beamSize,Nspec,bandwidth,calcThermalLevels=False):
    pos_center,vel_center,Lhat,r0,orientation_maxima = ReadStats(statsDir);

    mask,impact,distance,pos_observer,vel_observer,sightline = ReadSightlineFile(sightlineDir,tid) #Get sightline information. Mask limits snapshot to only intersecting particles to reduce load time

    gPos,gVel,gMass,gKernal,gTemp =  LoadDataForSightlineIteration(snapdir,Nsnapstring,0,mask,pos_center,vel_center) #Load particles only on sightline
    gPos,gVel = OrientGalaxy(gPos,gVel,Lhat,r0) #Orient Galaxy
    gDoppler = GetRadialVelocity(gPos-pos_observer,gVel-vel_observer) #Get radial velocity towards observer

    speciesMassFrac = LoadSpeciesMassFrac(snapdir,Nsnapstring,0,mask,speciesToRun,Gmas=gMass,KernalLengths=gKernal) #May be necesarry to load entire metallicity entry? CHECK!!!!
    spectrum,emission,optical_depth,nu = GenerateSpectra(gMass,speciesMassFrac,gDoppler,gKernal,gTemp,distance,impact,species=speciesToRun,beamSize=beamSize,Nspec=Nspec,bandwidth=bandwidth,calcThermalLevels=calcThermalLevels)

    return emission

