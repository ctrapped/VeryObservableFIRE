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



def BinData3d(data, sample, Nbins, operation = 'sum', weight = None):
    max_r = np.max(sample[:,0])
    nr,nTheta,nPhi = Nbins
    if weight is not None:
        data = np.multiply(data,weight)
        binned_weight_3d,tmp,tmp = stats.binned_statistic_dd(sample,weight,'sum',[nr,nTheta,nPhi],range=[[0,max_r],[-pi/2,pi/2],[-pi,pi]])

    binned_data_3d,binedge,binnum = stats.binned_statistic_dd(sample,data,'sum',[nr,nTheta,nPhi],range=[[0,max_r],[-pi/2,pi/2],[-pi,pi]])

    if weight is not None and operation=='mean':
        binned_weight_3d[binned_weight_3d==0]=eps;
        binned_data_3d = np.divide(binned_data_3d , binned_weight_3d)
    
    return binned_data_3d

    

def FlattenBinnedData(binned_data_3d,rRes,operation='sum',weight=None,axis=0):
    if weight is not None:
        binned_data_3d = np.multiply(binned_data_3d,weight)

    peakRadii = np.argmax(binned_data_3d,axis) * rRes

    nR,nTheta,nPhi = np.shape(binned_data_3d)
    radius_mat = np.zeros((nR,nTheta,nPhi))
    for ii in range(0,nR):
        radius_mat[ii,:,:] = ii*rRes
    weighted_radius_mat = np.multiply(radius_mat,binned_data_3d)
    meanRadii = np.divide(np.sum(weighted_radius_mat,axis) , np.sum(binned_data_3d,axis))


    
    if operation=='sum':
        binned_data_2d = np.sum(binned_data_3d,axis)
    elif operation=='mean' and weight is not None:
        binned_data_2d = np.divide(np.sum(binned_data_3d,axis) , np.sum(weight,axis))
    elif operation=='mean':
        binned_data_2d = np.mean(binned_data_3d,axis)


    return binned_data_2d, peakRadii, meanRadii
   


def VectorArrayMag(r):
    r_magnitude = np.sqrt(np.add(np.power(r[:,0],2),np.add(np.power(r[:,1],2),np.power(r[:,2],2))))
    return r_magnitude







