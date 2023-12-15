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

####Function to calculate the rotational velocity curve of a galaxy. Used for placing a mock observer within a galaxy
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023


def FindRotationCurve(pos,vel,mass,nr,max_r):
    rMag = VectorArrayMag(pos)
    rHat = np.copy(pos)
    rHat[:,0] = np.divide(pos[:,0],rMag)
    rHat[:,1] = np.divide(pos[:,1],rMag)
    rHat[:,2] = np.divide(pos[:,2],rMag)
    vMag = VectorArrayMag(vel - np.multiply(vel,rHat)) #Don't count radial velocity

    #Calculate mass weighted average
    binned_mom,binedge1d,binnum1d = stats.binned_statistic_dd(rMag,np.multiply(vMag,mass),'sum',nr,range=[[0,max_r]])
    binned_mass,binedge1d,binnum1d = stats.binned_statistic_dd(rMag,mass,'sum',nr,range=[[0,max_r]])

    return np.divide(binned_mom,binned_mass)

   

def VectorArrayMag(r):
    r_magnitude = np.sqrt(np.add(np.power(r[:,0],2),np.add(np.power(r[:,1],2),np.power(r[:,2],2))))
    return r_magnitude






