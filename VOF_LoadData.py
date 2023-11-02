import numpy as np
import h5py as h5py
import os.path
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import math
import sys
import time
import scipy.stats as stats
from VeryObservableFIRE.VOF_readsnap import *
#from build_shieldLengths import FindShieldLength #From Matt Orr


unit_M = 10**10 * 1.98855 *10**33 #10^10 solar masses / h !!in grams!! #h accounted for in readsnap
unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
unit_V = 1.0*10.0**5 # 1 km/s !!in cm/s!! Converted by factor of sqrt(a) in readsnap
unit_T = unit_L/unit_V
unit_rho = unit_M / unit_L**3

proton_mass = 1.6726219*10**(-27)*(1000.0/unit_M) ##appropriate units

pi = np.pi
eps = 0.00000000000000000000000000000000000000000000001


def ReadStats(statsDir):
    hf = h5py.File(statsDir,'r')
    posCenter = np.zeros((3))
    velCenter = np.zeros((3))
    Lhat = np.zeros((3))
    r0 = np.zeros((3))
    orientation_maxima = np.zeros((3))

    posCenter[:] = hf['pos_center'][:]
    velCenter[:] = hf['vel_center'][:]
    Lhat[:] = hf['Lhat'][:]
    r0[:] = hf['r0'][:]
    orientation_maxima[0] = np.array(hf['max_x'])
    orientation_maxima[1] = np.array(hf['max_y'])
    orientation_maxima[2] = np.array(hf['max_z'])

    hf.close()
    return posCenter,velCenter,Lhat,r0,orientation_maxima

def ReadSightlineFile(sightlineDir,tid):
    hf = h5py.File(sightlineDir,'r')
    groupName = "sightline"+str(tid)
    mask = np.array(hf[groupName].get('mask'))
    impact = np.array(hf[groupName].get('impact'))
    distance = np.array(hf[groupName].get('distance'))
    pos_observer=np.array(hf[groupName].get('pos_observer'))
    vel_observer=np.array(hf[groupName].get('vel_observer'))
    sightline = np.array(hf[groupName].get('sightline'))

    return mask,impact,distance,pos_observer,vel_observer,sightline


def LoadData(snapdir,Nsnapstring,ptype,rTrunc,posCenter,velCenter,buildShieldLengths=False):
    #Load position and density to create a mask for particles in user defined region of interest
    particles = readsnap_initial(snapdir, Nsnapstring, ptype, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) ##!! EDIT READSNAP INITIAL TO ONLY LOAD POS
    pos = particles['p'] #positions

    truncMax = posCenter + rTrunc
    truncMin = posCenter - rTrunc

    truncMask =  (pos[:,0]<truncMax[0]) & (pos[:,0] > truncMin[0]) & (pos[:,1] < truncMax[1]) & (pos[:,1] > truncMin[1]) & (pos[:,2] < truncMax[2]) & (pos[:,2] > truncMin[2])

    #Option, build shield lengths for each particle based on neighboring particles. Takes a lot of time 
    #NEED TO CONVERT TO P3
    #if buildShieldLengths:
    #    sph_shieldLength = FindShieldLength(3.086e21*Gpos,404.3*Gdens*0.76) # output units: cm.
    #    sph_shieldLength /= unit_L #convert to sim units
    #    sph_shieldLength = sph_shieldLength[truncMask] #Truncate after to ensure all shielding gas accounted for

   
    pos = pos[truncMask]

    #Load the rest of the data only in the ROI to save memory
    particles = readsnap_trunc(snapdir, Nsnapstring, 0, truncMask, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) ##!! EDIT TO LOAD DENS AS WELL
    vel = particles['v']
    mass = particles['m']

    #Center the Data on galactic center
    pos -= posCenter
    vel -= velCenter

    if (ptype==0):
        dens = particles['rho']
        metallicity = particles['z']
        neutral_H_frac = particles['nh']
        kernal_lengths = particles['h']
        temp = calcTemps(particles['u'],particles['ne'],metallicity)

        return pos,vel,mass,dens,metallicity,neutral_H_frac,kernal_lengths,temp

    else:
        return pos,vel,mass
        

def calcTemps(GintE,ElectronAbundance,Gz): #Calculate Temperatures
    gamma = 1.6666666666666666667
    kb =  1.38064852*10**(-23)*(100.0/unit_L)**2*(1000.0/unit_M)*(unit_T)**2 ## Boltzmann's constant, appropriate units
    #GintE = G['u'] #internal energy
    #ElectronAbundance = G['ne']
    #Gz = G['z']
    N = np.size(GintE)
    
    if (np.size(np.shape(Gz))>1):
        helium_mass_fraction=np.zeros((N))
        helium_mass_fraction[:] = Gz[:,1]
    else:
        helium_mass_fraction = Gz

    y_helium = np.divide(helium_mass_fraction,np.multiply(4,np.subtract(1,helium_mass_fraction)))
    #y_helium = helium_mass_fraction / (4*(1-helium_mass_fraction))
    mu=np.divide(np.add(1,np.multiply(4,y_helium)),np.add(1,np.add(y_helium,ElectronAbundance)))
    #mu = (1 + 4*y_helium) / (1+y_helium+ElectronAbundance)
    mean_molecular_weight = np.multiply(mu,proton_mass)
    #mean_molecular_weight = mu*proton_mass
    Gtemp = np.multiply(mean_molecular_weight,np.multiply(((gamma-1)/kb),GintE))
    #Gtemp = mean_molecular_weight * (gamma-1) * GintE / kb

    return Gtemp


def LoadDataForSightlineGenerator(snapdir,Nsnapstring,ptype,rTrunc,posCenter,velCenter,buildShieldLengths=False):
    #Load position and density to create a mask for particles in user defined region of interest
    particles = readsnap_initial(snapdir, Nsnapstring, ptype, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) ##!! EDIT READSNAP INITIAL TO ONLY LOAD POS
    pos = particles['p'] #positions

    #truncMax = posCenter + rTrunc
    #truncMin = posCenter - rTrunc

    #truncMask =  (pos[:,0]<truncMax[0]) & (pos[:,0] > truncMin[0]) & (pos[:,1] < truncMax[1]) & (pos[:,1] > truncMin[1]) & (pos[:,2] < truncMax[2]) & (pos[:,2] > truncMin[2])

    #Option, build shield lengths for each particle based on neighboring particles. Takes a lot of time
    if buildShieldLengths:
        sph_shieldLength = FindShieldLength(3.086e21*Gpos,404.3*Gdens*0.76) # output units: cm.
        sph_shieldLength /= unit_L #convert to sim units
        #sph_shieldLength = sph_shieldLength[truncMask] #Truncate after to ensure all shielding gas accounted for

   
    #pos = pos[truncMask]

    #Load the rest of the data only in the ROI to save memory
    particles = readsnap_sightline_gen(snapdir, Nsnapstring, 0, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1)
    #vel = particles['v']
    #mass = particles['m']

    #Center the Data on galactic center
    pos -= posCenter
    #vel -= velCenter

    if (ptype==0):
    #    dens = particles['rho']
    #    metallicity = particles['z']
    #    neutral_H_frac = particles['nh']
        kernal_lengths = particles['h']
        pid = particles['pid']
        cid = particles['cid']
        gen = particles['gen']
    #    temp = calcTemps(particles['u'],particles['ne'],metallicity)

        return pos,kernal_lengths,pid,cid,gen

    else:
        return pos
        
def LoadDataForHVCSightlineGenerator(snapdir,Nsnapstring,ptype,rTrunc,posCenter,velCenter,buildShieldLengths=False):
    #Load position and density to create a mask for particles in user defined region of interest
    particles = readsnap_initial(snapdir, Nsnapstring, ptype, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) ##!! EDIT READSNAP INITIAL TO ONLY LOAD POS
    pos = particles['p'] #positions

    truncMax = posCenter + rTrunc
    truncMin = posCenter - rTrunc

    truncMask =  (pos[:,0]<truncMax[0]) & (pos[:,0] > truncMin[0]) & (pos[:,1] < truncMax[1]) & (pos[:,1] > truncMin[1]) & (pos[:,2] < truncMax[2]) & (pos[:,2] > truncMin[2])

    #Option, build shield lengths for each particle based on neighboring particles. Takes a lot of time
    if buildShieldLengths:
        sph_shieldLength = FindShieldLength(3.086e21*Gpos,404.3*Gdens*0.76) # output units: cm.
        sph_shieldLength /= unit_L #convert to sim units
        sph_shieldLength = sph_shieldLength[truncMask] #Truncate after to ensure all shielding gas accounted for

   
    pos = pos[truncMask]

    #Load the rest of the data only in the ROI to save memory
    particles = readsnap_trunc(snapdir, Nsnapstring, 0, truncMask, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) ##!! EDIT TO LOAD DENS AS WELL
    vel = particles['v']
    mass = particles['m']


    #Center the Data on galactic center
    pos -= posCenter
    vel -= velCenter

    if (ptype==0):
        kernal_lengths = particles['h']
        pid = particles['pid']
        cid = particles['cid']
        gen = particles['gen']
        return pos,vel,mass,kernal_lengths,pid,cid,gen

    else:
        return pos,vel,mass
        
        
def LoadDataForFirelineIntersection(snapdir,Nsnapstring,ptype=0,buildShieldLengths=False):
    #Load position and density to create a mask for particles in user defined region of interest

    #Load the rest of the data only in the ROI to save memory
    particles = readsnap_firelineIntersection(snapdir, Nsnapstring, 0, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) ##!! EDIT TO LOAD DENS AS WELL



    pos = particles['p'] #positions

    if (ptype==0):
        pid = particles['pid']
        cid = particles['cid']
        gen = particles['gen']
        return pos,pid,cid,gen

    else:
        return pos

def LoadDataForSightlineIteration(snapdir,Nsnapstring,ptype,mask,pos_center,vel_center,buildShieldLengths=False):
    if mask is None:
        particles = readsnap_sightline_itr(snapdir, Nsnapstring, ptype, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1)
    else:
        particles = readsnap_trunc_sightline_itr(snapdir, Nsnapstring, ptype, mask, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1)
    
    temp = calcTemps(particles['u'],particles['ne'],particles['z']) ##SEE IF YOU CAN ONLY LOAD IN 1 METALLICITY FILE AT A TIME


    return particles['p']-pos_center, particles['v']-vel_center, particles['m'], particles['h'], temp



def LoadSpeciesMassFrac(snapdir,Nsnapstring,ptype,mask,species,buildShieldLengths=False,Gmas=None,KernalLengths=None):
    if mask is None:
        p = readsnap_speciesMassFrac_noMask(snapdir,Nsnapstring,ptype,species,snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1)
    else:
        p = readsnap_speciesMassFrac(snapdir,Nsnapstring,ptype,mask,species,snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1)
    if (species=="h_alpha"):
        #need to be passed mass, kernal
        #need to load Gnh,density,Gz[:,0],Gz[:,1]
        return CalcMolecularFraction(p['nh'],p['hsml'],p['rho'],p['fHe'],p['z'],to_return="fH1")
    if (species=="HI_21cm"):
        return CalcMolecularFraction(p['nh'],p['hsml'],p['rho'],p['fHe'],p['z'],to_return="fH1")



def CalcMolecularFraction(Gnh,KernalLengths,density,fHe,fMetals,to_return="fH1",buildShieldLengths=False,sph_shieldLength=None):
    #Inputs = Mass, number density, Kernal Length, density, metallicity
    Z = fMetals #metal mass (everything not H, He)
    M_H = 1.67353269159582103*10**(-27)*(1000.0/unit_M) ##kg to g to unit mass
    mu_H = 2.3*np.power(10.0,-27.0)*(1000.0/unit_M) #'average' mass of hydrogen nucleus from Krumholz&Gnedin 2011

    if (to_return=="fHion"):
        return (1. - fHe - fMetals) * (1.-Gnh)

    Z_MW = 0.02 #Assuming MW metallicity is ~solar on average (Check!!)
    Z_solar = 0.02 #From Gizmo documentation

    N_ngb = 32.
    if buildShieldLengths:
        sobColDens = np.multiply(sph_shieldLength,density) + np.multiply(KernalLengths,density) / np.power(N_ngb,1./3.) ## kernal Length/(nearest neighbor number)^{1/3{
    else:
        sobColDens =np.multiply(KernalLengths,density) / np.power(N_ngb,1./3.) #Cheesy approximation of Column density
    tau = np.multiply(sobColDens,Z)*1/(mu_H*Z_MW) * np.power(10.0,-21.0)/(unit_L**(2)) #cm^2 to Unit_L^2
    tau[tau==0]=eps #avoid divide by 0

    chi = 3.1 * (1+3.1*np.power(Z/Z_solar,0.365)) / 4.1 #Approximation

    s = np.divide( np.log(1+0.6*chi+0.01*np.power(chi,2)) , (0.6 *tau) )
    s[s==-4.] = -4.+eps #Avoid divide by zero
    fH2 = np.divide((1 - 0.5*s) , (1+0.25*s)) #Fraction of Molecular Hydrogen from Krumholz & Knedin
    
    lowerApproxLimit = 0.1
    Zprime = np.divide(Z,Z_solar)
    sigma0 = sobColDens * np.power(10.0,4.0) # Surface column density to Msolar/pc^2
    fc = np.divide(tau , 0.066*Zprime*sigma0)# should be~1
    SFR_0 = 0.25*np.power(10.0,-3.0) #Unit_m * kpc^-2 * Gyr -1
   # denom = np.multiply( sobColDens , np.multiply(2.3*np.multiply(fc,Zprime) , np.divide(4.1,1+3.1*np.power(Zprime,0.365))))
    denom = 7.2*tau
    denom[denom==0]=eps
    fH2[fH2<lowerApproxLimit] = 1.0/3.0 * ( 2.0 - 44.0*2.5*0.066/7.2 * np.divide(chi[fH2<lowerApproxLimit],tau[fH2<lowerApproxLimit]))
    
    
    fH2[fH2<0] = 0 #Nonphysical negative molecular fractions set to 0
    
    #if(np.size(fHe>0)):
    #    print("f_hydrogen=",np.min(1.-fHe - fMetals),np.max(1.-fHe - fMetals))
    #    print("f_h_neutral=",np.min(Gnh),np.max(Gnh))
    #    print("max f_h2=",np.max(fH2))
    #    print("f_h_i=",np.min(1.-fH2-1+Gnh),np.max(1.-fH2-1+Gnh))
    
    #fHion = 1.-Gnh
    #fHI = 1 - fH2 - fHion
    #print(np.size(fHI[fHI<0]),"fHI entries less than 0?")
    #fHI[fHI<0]=0
    
    if (to_return=="fH1"):
        fH1 = np.multiply((1. - fHe - fMetals) , (1. - fH2 - (1. - Gnh)))
        fH1[fH1<0]=0
        return fH1
    elif (to_return=="fH2"):
        return (1. - fHe - fMetals) * fH2

    
    #return NH1,NH2,NHion

def VectorArrayMag(r):
    r_magnitude = np.sqrt(np.add(np.power(r[:,0],2),np.add(np.power(r[:,1],2),np.power(r[:,2],2))))
    return r_magnitude










