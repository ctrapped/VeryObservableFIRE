import numpy as np
import h5py as h5py
import math
import time

import scipy
from scipy.sparse import csr_matrix
from scipy.signal import oaconvolve
from scipy.sparse import coo_matrix, csr_matrix
from scipy.fft import rfft2, irfft2, next_fast_len

from LoadData import ReadStats
from LoadData import LoadDataForImageGen
from LoadData import LoadMaskedDataForImageGen
from LoadData import LoadData
from GenerateSpectra import GenerateSpectra

from CalcRotationCurve import CalcRotationCurve
from OrientGalaxy import CenterOnObserver
from OrientGalaxy import OrientGalaxy
from CalculateSightlineSpectrum import CalculateSightlineSpectrum

from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool

from functools import partial


unit_M = 10**10 * 1.98855 *10**33 #10^10 solar masses / h !!in grams!! #h accounted for in readsnap
unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
unit_V = 1.0*10.0**5 # 1 km/s !!in cm/s!! Converted by factor of sqrt(a) in readsnap
unit_T = unit_L/unit_V
unit_rho = unit_M / unit_L**3

Joules2eV = 6.241509*np.power(10.0,18.0)
meters2Kpc = 3.24078*np.power(10.0,-20.0)
Jy2SimUnits = np.power(10.0,-26.0) * Joules2eV / meters2Kpc / meters2Kpc
#J*s^-1*m^-2*Hz-1
SimUnits2Jy = 1.0 / Jy2SimUnits 

proton_mass = 1.6726219*10**(-27)*(1000.0/unit_M) ##appropriate units

pi = np.pi

arcsec2rad = pi / (180*3600)
eps = 1e-20

####Functions to generate a sightline spectra for a given set of observer parameters. Defines the vectors, overlapping particles in each sightline,
####and relevant parameters that allows each thread to get the effective column density of a particle along that sightline.
####
####Then calculates the emission and absorption from each particle along the sightline.
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 03-10-2023

def GenSightline(thread_id,Nsightlines_1d,sightlines,gPos,gVel,gKernel,gMas,gTemp,speciesMassFrac,beamSize,speciesToRun,Nspec,bandwidth,rObserver):
    ix=thread_id % Nsightlines_1d #Thread pixel coordinates
    iy = int(np.floor(thread_id / Nsightlines_1d))
    t0=time.time()
    spectrum,emission=CalculateSightlineSpectrum(sightlines[ix,iy,:],gPos,gVel,gKernel,gMas,gTemp,speciesMassFrac,beamSize,speciesToRun,Nspec,bandwidth,rObserver,calcThermalLevels=False)
    print("Finished sightline [",ix,',',iy,'] in ',time.time()-t0)
    return emission#, [ix,iy]



def sphere_kernel(d):
    #Kernel for smoothing as particles by their smoothing length
    if d == 0:
        return np.ones((1, 1))
    c = np.arange(-d, d + 1)
    return np.sqrt(np.clip(d**2 - (c[:, None]**2 + c[None, :]**2), 0, None)) / d



def DepositParticles(gPos,gVel,gKernel,gMas,gTemp,speciesMassFrac,Nsightlines_1d,beamSize,speciesToRun,Nspec,bandwidth,rObserver,Lhat,r_0,max_r,projectGasProperties=False,gRmom=None,gSmom=None,gRotMom=None):
    t0=time.time()

    pixel_size_physical = 2*max_r/Nsightlines_1d

    #Already rotated
    zmag = gPos[:,0]
    xmag = gPos[:,2]
    ymag = gPos[:,1]

    N,dim = np.shape(gPos)
    impact = np.zeros_like(gMas) #Assume dead center, will be smoothed later

    fx = (xmag / max_r + 1) / 2 * Nsightlines_1d
    fy = (ymag / max_r + 1) / 2 * Nsightlines_1d
    pixel_coords_x = np.floor(fx).astype(int)
    pixel_coords_y = np.floor(fy).astype(int)
    mask = (pixel_coords_x >= 0) & (pixel_coords_x < Nsightlines_1d) & \
       (pixel_coords_y >= 0) & (pixel_coords_y < Nsightlines_1d)

    dopplerVelocity = gVel[mask][:,0]
    dopplerVelocity = np.add(dopplerVelocity , (zmag[mask]-rObserver) * 0.07) #kpc * km/s /kpc, Hubble flow but still centered on galaxy

    distance = np.copy(zmag[mask])
    distance[distance==0]=eps

    spectrum,emission,tau,nu = GenerateSpectra(gMas[mask],speciesMassFrac[mask],dopplerVelocity,gKernel[mask],gTemp[mask],distance,impact[mask],speciesToRun,beamSize,Nspec,bandwidth,calcThermalLevels=False,calcChordLength=False,return_sightline=False)

    N = Nsightlines_1d
    Npix = N * N
    emission_map = np.zeros((N, N, Nspec))
    if projectGasProperties:
        mass_map = np.zeros((N,N))
        rMom_map = np.copy(mass_map)
        sMom_map = np.copy(mass_map)
        rotMom_map = np.copy(mass_map)


    px = pixel_coords_x[mask]
    py = pixel_coords_y[mask]
    dx = np.round(gKernel[mask] / pixel_size_physical - 0.5).astype(np.int64) #pixelized smoothing

    order = np.argsort(dx, kind='stable')
    dx_s, px_s, py_s = dx[order], px[order], py[order]
    em_s = emission[order]

    if projectGasProperties:
        props_s = np.stack([gMas[mask], gRmom[mask], gSmom[mask], gRotMom[mask]], axis=1)[order] #properties to project

    dx_vals, starts, counts = np.unique(dx_s, return_index=True, return_counts=True) #Unique smoothing groups

    stamp_threshold = 5 * Npix
    rows_l, cols_l, vals_l = [], [], []
    conv_groups = []

    for d, s, n in zip(dx_vals, starts, counts):
        k = sphere_kernel(d)
        if projectGasProperties:
            props_s[s:s+n] /= k.sum()          # per-group: weights for properties sum to 1
        ox, oy = np.nonzero(k)
        w = k[ox, oy]
        if n * w.size < stamp_threshold:
            X = px_s[s:s+n, None] + (ox - d)
            Y = py_s[s:s+n, None] + (oy - d)
            ok = (X >= 0) & (X < N) & (Y >= 0) & (Y < N)
            rows_l.append((X * N + Y)[ok])
            cols_l.append(np.broadcast_to(np.arange(s, s + n)[:, None], X.shape)[ok])
            vals_l.append(np.broadcast_to(w, X.shape)[ok])
        else:
            conv_groups.append((d, s, n, k))

    # Properties become extra channels after the per-group scaling above
    if projectGasProperties:
        em_s = np.concatenate([em_s, props_s], axis=1)
    Nch = em_s.shape[1]
    out_map = np.zeros((N, N, Nch))

    if rows_l:
        P = coo_matrix((np.concatenate(vals_l),
                        (np.concatenate(rows_l), np.concatenate(cols_l))),
                       shape=(Npix, len(dx_s))).tocsr()
        out_map += (P @ em_s).reshape(N, N, Nch)

    if conv_groups:
        dmax = max(g[0] for g in conv_groups)
        L = next_fast_len(N + dmax, real=True)
        acc = None
        for d, s, n, k in conv_groups:
            flat = px_s[s:s+n] * N + py_s[s:s+n]
            Pg = csr_matrix((np.ones(n), (flat, np.arange(n))), shape=(Npix, n))
            tmp = (Pg @ em_s[s:s+n]).reshape(N, N, Nch)

            kp = np.zeros((L, L))
            kp[:2*d+1, :2*d+1] = k
            kp = np.roll(kp, (-d, -d), axis=(0, 1))

            F = rfft2(tmp, s=(L, L), axes=(0, 1), workers=-1)
            F *= rfft2(kp)[:, :, None]
            if acc is None:
                acc = F
            else:
                acc += F
        out_map += irfft2(acc, s=(L, L), axes=(0, 1), workers=-1)[:N, :N]

    emission_map = out_map[..., :Nspec]
    if projectGasProperties:
        mass_map, rMom_map, sMom_map, rotMom_map = np.moveaxis(out_map[..., Nspec:], -1, 0)
        return emission_map, mass_map, rMom_map, sMom_map, rotMom_map
    return emission_map


def ProjectImage(config,snapdir,Nsnapstring,statsDir,observer_position,maxima,Nsightlines,inclination,position_angle,particleData=None,sightlines=None):
    #Config values this function (and the DepositParticles/GenSightline/CalculateSightlineSpectrum/GenerateSpectra
    #helpers it calls) need, read out of the single shared config dict.
    observer_velocity = config['observerVelocity']
    beamSize = config['beamSize']
    phiObs = config['phiObs']
    speciesToRun = config['speciesToRun']
    Nspec = config['Nspec']
    bandwidth = config['bandwidth']
    targetBeamSize = config['targetBeamSize']
    noiseAmplitude = config['noiseAmplitude']
    num_cores = config['num_cores']
    projectGasProperties = config['projectGasProperties']
    runAsOpticallyThinProjection = config['runAsOpticallyThinProjection']

    max_r,maxPhi,maxTheta = maxima
    rObserver=np.abs(observer_position[0])
    pos_center,vel_center,Lhat,r0,orientation_maxima = ReadStats(statsDir)
    
    if inclination>0:
        #Rotate r0 first to set the position angle
        rotation_axis = np.copy(Lhat)
        rotation_vector = float(position_angle)*np.pi/180.*rotation_axis
        rotation = scipy.spatial.transform.Rotation.from_rotvec(rotation_vector)
        r0 = rotation.apply(r0)
        
        #Rotate the vectors Lhat and r0 that define the z and x unit vectors respectively. Effectively rotates the entire galaxy
        rotation_axis = np.cross(Lhat,r0)
        rotation_vector = float(inclination)*np.pi/180.*rotation_axis
        rotation = scipy.spatial.transform.Rotation.from_rotvec(rotation_vector)

        Lhat = rotation.apply(Lhat)
        r0 =   rotation.apply(r0)


    gPos = particleData['pos']
    gVel = particleData['vel']
    gKernel = particleData['kernel']
    gMas=particleData['mass']
    gTemp=particleData['temp']
    speciesMassFrac=particleData['speciesMassFrac']
    if projectGasProperties:
        gRmom=particleData['rMom']
        gSmom=particleData['sMom']
        gRotMom = particleData['rotMom']

    #If observer velocity is not defined, calculate rotation curve to put the observer in the galaxy. Should only be used for in galaxy observations.
    rotationCurve=None
    defineRotationCurve=False
    
    if observer_velocity is None: defineRotationCurve=True
    
    if defineRotationCurve:
        #Load the star particles
        sPos,sVel,sMass = LoadData(snapdir,Nsnapstring,1,max_r,pos_center,vel_center)
        #Transform into previously defined coordinate system
        sPos,sVel = OrientGalaxy(sPos,sVel,Lhat,r0)

        rBinSize=0.1
        nr = int(math.ceil(max_r/rBinSize)) 
        rotationCurve = CalcRotationCurve(sPos,sVel,sMass,nr,max_r)


    pos_observer,vel_observer = CenterOnObserver(observer_position,observer_velocity,rotationCurve=rotationCurve,max_r=max_r)
    print("vel_observer=",vel_observer)
    gPos -= pos_observer #Switch to observers frame of reference
    gVel -= vel_observer

    Nsightlines_1d = int(np.round(np.sqrt(Nsightlines)))

    if num_cores is None: num_cores = multiprocessing.cpu_count()-1
    if num_cores > multiprocessing.cpu_count()-1: num_cores = multiprocessing.cpu_count()-1


    tStart=time.time()


    if runAsOpticallyThinProjection:
        Npart = np.size(gMas)
        ppt = int(np.ceil(Npart/num_cores))
        if projectGasProperties:
            image,mass_map,rMom_map,sMom_map,rotMom_map = DepositParticles(gPos,gVel,gKernel,gMas,gTemp,speciesMassFrac,Nsightlines_1d,beamSize,speciesToRun,Nspec,bandwidth,rObserver,Lhat,r0,max_r,projectGasProperties=projectGasProperties,gRmom=gRmom,gSmom=gSmom,gRotMom=gRotMom)
        else:
            image = DepositParticles(gPos,gVel,gKernel,gMas,gTemp,speciesMassFrac,Nsightlines_1d,beamSize,speciesToRun,Nspec,bandwidth,rObserver,Lhat,r0,max_r,projectGasProperties=projectGasProperties)

    else:
        if sightlines is None: #Create sightline vectors to evenly sample the observed space
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
        print("Working with ",num_cores," cores")

        #Define partial function to parallelize
        GenSightline_ = partial(GenSightline,Nsightlines_1d=Nsightlines_1d,sightlines=sightlines,gPos=gPos,gVel=gVel,gKernel=gKernel,gMas=gMas,gTemp=gTemp,speciesMassFrac=speciesMassFrac,beamSize=beamSize,speciesToRun=speciesToRun,Nspec=Nspec,bandwidth=bandwidth,rObserver=rObserver)

        print("Splitting into parallel runs")
        x= Parallel(n_jobs=num_cores)(delayed(GenSightline_)(i) for i in sightline_indices)

        print("Reshaping image...")
        image = np.reshape(x,[Nsightlines_1d,Nsightlines_1d,Nspec])


    tParallel=time.time()


    ###Smooth here
    obs_spatial_resolution = targetBeamSize * np.linalg.norm(observer_position)
    base_spatial_resolution = (2*maxima[0]) / Nsightlines_1d
    print("Obs spatial resolution (kpc)=",obs_spatial_resolution)
    print("Base spatial resolution (kpc)=",base_spatial_resolution)
    sigma = obs_spatial_resolution/base_spatial_resolution/ (2*np.sqrt(2*np.log(2)))
    smoothed_image = scipy.ndimage.gaussian_filter(image, sigma=sigma , axes=[0,1])

    ####Add noise here
    noiseProfile = np.random.normal(0, noiseAmplitude, np.shape(image)) #Create noise profile scaled by the downsampling we are doing
    noiseProfile = scipy.ndimage.gaussian_filter(noiseProfile, sigma = sigma, axes=[0,1])
    noiseProfile = noiseProfile * noiseAmplitude / np.std(noiseProfile) #Renormalize noise
    noisy_image = np.add(smoothed_image, noiseProfile)

    print("max of image is:",np.max(smoothed_image))
    print("Max of noise profile is:",np.max(noiseProfile))
    print("Time to run in parallel=",tParallel-tStart)
    print("Total Time for image generation=",time.time()-tStart)

    if projectGasProperties:
        return image, smoothed_image, noisy_image, mass_map,rMom_map,sMom_map,rotMom_map
    return image, smoothed_image, noisy_image
