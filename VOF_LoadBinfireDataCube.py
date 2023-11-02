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



def LoadBinfireDataCube(filedir,statsDir,Nsnapstring):
    hf_s = h5py.File(statsDir,'r')
    nx = int(np.round(2* np.array(hf_s['max_x']) / np.array(hf_s['xbin_size']) ))
    ny = int(np.round(2* np.array(hf_s['max_y']) / np.array(hf_s['ybin_size']) ))
    nz = int(np.round(2* np.array(hf_s['max_z']) / np.array(hf_s['zbin_size']) ))
    hf_s.close()
   
    hf = h5py.File(filedir+Nsnapstring.zfill(4)+".hdf5",'r')
    indices = hf['indices']

    mass = np.zeros((nx,ny,nz))
    mass[indices] = hf['mass']

    dens = np.zeros((nx,ny,nz))
    dens[indices] = hf['density']
    
    mom_s = np.zeros((nx,ny,nz))
    mom_s[indices] = hf['mom_s']

    mom_z = np.zeros((nx,ny,nz))
    mom_z[indices] = hf['mom_z']
    mom_z[:,:,0:int(nz/2)] *= (-1) #make all inflows negative

    return mass,dens,mom_s,mom_z
