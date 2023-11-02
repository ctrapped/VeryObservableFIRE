import numpy as np
import h5py as h5py
import os.path
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import math
import sys
import time
import scipy.stats as stats

from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool

from functools import partial

from VeryObservableFIRE.VeryObservableFIRE import ConvolveWithPSF

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


pi = np.pi
arcsec = (1. /60. / 60.) * pi/180.

image=0



def AddNoiseAndPSF(image , noiseAmplitude):
    noiseProfile = np.random.normal(0, noiseAmplitude, np.shape(image))
    image = np.add(image,noiseProfile)
    newImage = np.zeros(np.shape(image))

    for ix in range(0,np.shape(image)[0]):
        for iy in range(0,np.shape(image)[1]):
            newImage = np.add(newImage , ConvolveWithPSF(image,ix,iy))
            
    return newImage



hfi = h5py.File(inputImageDir,'r')
inputImage = np.array(hfi['spectra'])

noiseAmplitude = 3 #Jy
outputImage = AddNoiseAndPSF(inputImage , noiseAmplitude)

hfo = h5py.File(outputImageDir,'w')
hfo.create_dataset('spectra',data=outputImage)

