import numpy as np
import h5py as h5py
import os.path
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import math
import sys
import time
import scipy
import scipy.stats as stats
from Binfire.readsnap_binfire import *
from Binfire.shrinking_sphere import *
#from Binfire.build_shieldLengths import FindShieldLength #From Matt Orr
from VeryObservableFIRE.VOF_OrientGalaxy import GetRadialVelocity
from VeryObservableFIRE.VOF_LoadData import ReadSightlineFile

unit_M = 10**10 * 1.98855 *10**33 #10^10 solar masses / h !!in grams!! #h accounted for in readsnap
unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
unit_V = 1.0*10.0**5 # 1 km/s !!in cm/s!! Converted by factor of sqrt(a) in readsnap
unit_T = unit_L/unit_V
unit_rho = unit_M / unit_L**3

proton_mass = 1.6726219*10**(-27)*(1000.0/unit_M) ##appropriate units

pi = 3.14159265359
eps = 0.00000000000000000000000000000000000000000000001

buildShieldLengths=False


#def RunBinfire(snapdir,Nsnapstring,output,suffix,maxima,res,tempMin,tempMax,densMin,densMax,phasetag,Nmetals,writeFlags,r_0=None,shrinking_sphere_flag=1,pos_center=None,Lhat=None,vel_center=None,time_array=None,old_a=None,old_v=None,rshrinksphere=5000,rminsphere=10,shrinkfactor=0.7,mask_center=None,mask_radius=None):

def RunBinfire(snapdir,statsDir,Nsnap,output,maxima,Nbins,tempMin=[None],tempMax=[None],densMin=[None],densMax=[None],phasetag=['AG'],rshrinksphere=5000,rminsphere=10,shrinkfactor=0.7,returnTiltedRingData=False,inclination=None):


    Nsnapstring = str(Nsnap)
    shrinking_sphere_flag=0
    writeStatsFile = False
     
    if statsDir is not None:
        try:
            r_0,pos_center,Lhat,vel_center = ReadStats(statsDir)
        except:
            writeStatsFile = True
            r_0=None
            pos_center=None
            vel_center=None
            Lhat=None
            shrinking_sphere_flag=1
    
    sph_shieldLength = None   
    t0 = time.time() #for keeping track of efficiency
    #Resolution Information
    max_x = np.pi/2*maxima[0];max_y= np.pi/2*maxima[1];max_vel=maxima[2]/2.
    max_z = np.pi/2*maxima[2]
    #xbin_size = res[0];ybin_size=res[1];zbin_size=res[2]
    
    nx,ny,nvel = Nbins
    

    #nz = int(math.ceil(2*max_z/zbin_size))
    #nvel = Nspec
    #nx = int(math.ceil(2*max_x/xbin_size))
   # ny = int(math.ceil(2*max_y/ybin_size))

    if np.size(np.shape(densMin))==0:
        densMin=[densMin]
    if np.size(np.shape(densMax))==0:
        densMax=[densMax]
    if np.size(np.shape(tempMin))==0:
        tempMin=[tempMin]
    if np.size(np.shape(tempMax))==0:
        tempMax=[tempMax]


    needToCenter = False
    if ((pos_center is None) or (vel_center is None)):
        needToCenter = True

    #Read in Header and Data
    print('Snapdir:',snapdir)
    header = readsnap_initial(snapdir, Nsnapstring,0,snapshot_name='snapshot',extension='.hdf5',h0=1,cosmological=1, header_only=1)
    ascale = header['time']
    h = header['hubble']

    #Read the Snapshots#################################################
    #Already accounts for factors of h, but not the hubble flow
    Gloaded = False;

    if (needToCenter):
        G = readsnap_initial(snapdir, Nsnapstring, 0, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Gas, only load position and density
        Gpos = G['p'] #positions
        Gdens = G['rho'] #Densities for finding center


        #if buildShieldLengths:
        #    t_shield = time.time()
        #    sph_shieldLength = FindShieldLength(3.086e21*Gpos,404.3*Gdens*0.76) # output units: cm.
        #    sph_shieldLength /= unit_L #convert to sim units
         #   print("Time to calculate shield lengths:", time.time()-t_shield)

        if pos_center is None:
          tmp_index = np.argmax(Gdens) 
          if not np.isscalar(tmp_index):
              center_index = tmp_index[0]
              print("Warning: ",np.size(tmp_index)," multiple max density particles")
          else:
              center_index = tmp_index #in case there are degenerate max densities

          pos_center = Gpos[center_index,:] #Naive center of sim.
          print("Center estimate from density: ",pos_center)


    if shrinking_sphere_flag:
        rTrunc = rshrinksphere #Only load relevant data
    else:
        rTrunc = np.sqrt(max_x*max_x + max_y*max_y)#max(max_y,max_x)
    truncMax = np.zeros((3))
    truncMin = np.zeros((3))
    truncMax[0] = pos_center[0]+rTrunc;truncMax[1] = pos_center[1]+rTrunc;truncMax[2] = pos_center[2]+rTrunc
    truncMin[0] = pos_center[0]-rTrunc;truncMin[1] = pos_center[1]-rTrunc;truncMin[2] = pos_center[2]-rTrunc

    Gloaded=False
    Sloaded=False
    DMloaded=False


    if True:
        if not needToCenter:
            G = readsnap_initial(snapdir, Nsnapstring, 0, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Gas, only load position and density
            Gpos = G['p'] #positions
            Gdens = G['rho'] #Densities for finding center

            #if buildShieldLengths:
            #    sph_shieldLength = FindShieldLength(3.086e21*Gpos,404.3*Gdens*0.76) # output units: cm.
            #    sph_shieldLength /= unit_L #convert to sim units

        t1 = time.time()
        truncMask =  (Gpos[:,0]<truncMax[0]) & (Gpos[:,0] > truncMin[0]) & (Gpos[:,1] < truncMax[1]) & (Gpos[:,1] > truncMin[1]) & (Gpos[:,2] < truncMax[2]) & (Gpos[:,2] > truncMin[2]) 
        Gpos = Gpos[truncMask]
        Gdens = Gdens[truncMask]
        #if buildShieldLengths:
        #    sph_shieldLength = sph_shieldLength[truncMask]

        G = readsnap_trunc(snapdir, Nsnapstring, 0, truncMask, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Gas, only load truncated data
        Gvel = G['v']#Velocities
        Gmas = G['m'] #masses
        Gz = G['z'] #metallicities
        Gnh = G['nh'] #neutral hydrogen
        Gh = G['h'] #Kernal Lengths for col dens approx
        Gtemp = calcTemps(G['u'],G['ne'],Gz)
        N = np.size(Gmas) #Number of Gas Particles
        Gloaded = True
        del G
        print("Time to load gas:",time.time()-t1)

    if (needToCenter):
        t1=time.time()
        DM = readsnap_initial(snapdir, Nsnapstring, 1, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Dark Matter, only load position
        DMpos = DM['p']
        truncMask = (DMpos[:,0]<truncMax[0]) & (DMpos[:,0] > truncMin[0]) & (DMpos[:,1] < truncMax[1]) & (DMpos[:,1] > truncMin[1]) & (DMpos[:,2] < truncMax[2]) & (DMpos[:,2] > truncMin[2]) 
        DMpos = DMpos[truncMask]

        DM = readsnap_trunc(snapdir, Nsnapstring, 1, truncMask, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Dark Matter, only load truncated data
        DMmas = DM['m']
        DMvel = DM['v']
        NDM = np.size(DMmas) #Number of Dark Matter Particles
        del DM
        DMloaded = True
        print("Time to load dark matter:",time.time()-t1)


        t1=time.time()
        S = readsnap_initial(snapdir, Nsnapstring, 4, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Stars, only load position
        Spos = S['p']
        truncMask =  ((Spos[:,0]<truncMax[0]) & (Spos[:,0] > truncMin[0]) & (Spos[:,1] < truncMax[1]) & (Spos[:,1] > truncMin[1]) & (Spos[:,2] < truncMax[2]) & (Spos[:,2] > truncMin[2])) 
        Spos = Spos[truncMask]

        S = readsnap_trunc(snapdir, Nsnapstring, 4, truncMask, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Stars Matter, only load truncated data
        if (needToCenter): #only need stellar velocities to determine galactic vel
            Svel = S['v']
        Smas = S['m']

        NSt = np.size(Smas) #Number of star particles
        del S
        del truncMask
        Sloaded = True
        print("Time to load stars:",time.time()-t1)

        t1=time.time()
        old_a = None
        old_v = None
        mask_center=None
        mask_radius=None
        
        pos_center = findCenter(pos_center,Gdens,Gpos,Spos,Smas,old_a,old_v,ascale,rshrinksphere,rminsphere,shrinkfactor,shrinking_sphere_flag,mask_center,mask_radius)
        print("Time to center:",time.time()-t1)

    if Gloaded:
       Gpos-=pos_center #Shift to Galactic Frame
       r2 = np.add(np.power(Gpos[:,0],2),np.add(np.power(Gpos[:,1],2),np.power(Gpos[:,2],2))) #find distance of particles from center
    if (DMloaded):
        r2_DM = np.add(np.power(np.subtract(DMpos[:,0],pos_center[0]),2),np.add(np.power(np.subtract(DMpos[:,1],pos_center[1]),2),np.power(np.subtract(DMpos[:,2],pos_center[2]),2)))
        del DMpos
    if (Sloaded):
        r2_S = np.add(np.power(np.subtract(Spos[:,0],pos_center[0]),2),np.add(np.power(np.subtract(Spos[:,1],pos_center[1]),2),np.power(np.subtract(Spos[:,2],pos_center[2]),2)))

    if (vel_center is None):
        t1=time.time()
        Sidx = (r2_S<15**2) #Find where the stars are within 15kpc of center to calculate velocity of center
        DMidx = (r2_DM<15**2)

        vel_center = center_of_mass_vel_gen(np.concatenate((Smas[Sidx],DMmas[DMidx]),0),np.concatenate((Svel[Sidx,:],DMvel[DMidx,:]),0))
        print("Time to find vel_center:",time.time()-t1)

    print("Best Estimate of Center Velocity:",vel_center)

    if Gloaded:
       Gvel-=vel_center
    if needToCenter:
        del Smas;del Svel;del DMvel;del r2_DM;del r2_S


#CALCULATE MOMENTUMS################################################################################

    Gmom = np.zeros((N,3));
    Gmom[:,0] = np.multiply(Gmas,Gvel[:,0]) #Calculate each Momentum component
    Gmom[:,1] = np.multiply(Gmas,Gvel[:,1])
    Gmom[:,2] = np.multiply(Gmas,Gvel[:,2])
    del Gvel #Delete for memory, work only in momentum from here out

    nh = findDensities(Gz,Gdens) #Calculate Density
    if (Lhat is None):
        t1=time.time()
        Lhat = findAngularMomentum(nh,Gpos,Gmom,Gtemp,r2) #Find orientation of the Disk
        print("Time to find Lhat:",time.time()-t1)



    t1=time.time()
    N = np.size(Gmas)


    r_z = np.zeros((N,3))
    r_s = np.zeros((N,3))



    smag = np.zeros((N))
    zmag = np.zeros((N))

    print("Finding velocity components...")

    zmag = np.dot(Gpos,Lhat)
    r_z[:,0] = zmag*Lhat[0]
    r_z[:,1] = zmag*Lhat[1]
    r_z[:,2] = zmag*Lhat[2]

    r_s = np.subtract(Gpos,r_z)
    smag = VectorArrayMag(r_s)
    smag[smag==0] = eps #make zero entries epsilon for division purposes


    N=np.size(smag) #Redefine number of particles we are looking at

    s_hat = np.zeros((N,3))
    r_hat = np.zeros((N,3))

    s_hat[:,0] = np.divide(r_s[:,0],smag)
    s_hat[:,1] = np.divide(r_s[:,1],smag)
    s_hat[:,2] = np.divide(r_s[:,2],smag)
    
    rmag = VectorArrayMag(Gpos)
    r_hat[:,0] = np.divide(Gpos[:,0],rmag)
    r_hat[:,1] = np.divide(Gpos[:,1],rmag)
    r_hat[:,2] = np.divide(Gpos[:,2],rmag)
    

    phi = np.zeros((N))
    if r_0 is None:
        print('Defining r0...')
        r_0 = np.zeros((3))
        r_0[:] = r_s[0,:] / np.linalg.norm(r_s[0,:])
    else:
        r_0 = r_0 - np.dot(r_0,Lhat)*Lhat #project onto current disk    
        r_0 = r_0 / np.linalg.norm(r_0) #make unit vector

    print('r0 is',r_0)
    r_0_forStats = np.copy(r_0)
    Lhat_forStats = np.copy(Lhat)
    
    Lhat=-Lhat

    
    if inclination is not None:
        rotation_axis = Lhat
        rotation_vector = np.pi*rotation_axis
        rotation = scipy.spatial.transform.Rotation.from_rotvec(rotation_vector)
    
        r_0 = rotation.apply(r_0)
        
    
    
        rotation_axis = np.cross(Lhat,r_0)
        rotation_vector = (90.-inclination)*np.pi/180.*rotation_axis
        rotation = scipy.spatial.transform.Rotation.from_rotvec(rotation_vector)
    
        Lhat = rotation.apply(Lhat)
        r_0 = rotation.apply(r_0)
        
        
        
        zmag = np.dot(Gpos,Lhat)
        r_z[:,0] = zmag*Lhat[0]
        r_z[:,1] = zmag*Lhat[1]
        r_z[:,2] = zmag*Lhat[2]

        r_s = np.subtract(Gpos,r_z)
        smag = VectorArrayMag(r_s)
        smag[smag==0] = eps #make zero entries epsilon for division purposes

        s_hat[:,0] = np.divide(r_s[:,0],smag)
        s_hat[:,1] = np.divide(r_s[:,1],smag)
        s_hat[:,2] = np.divide(r_s[:,2],smag)
        
        
        


    acos_term = np.divide(np.dot(r_s,r_0),(np.linalg.norm(r_0)*smag))
    acos_term[acos_term>1] = 1 #make sure the term isn't above magnitude 1 by a rounding error
    acos_term[acos_term<-1] = -1
    phi = np.multiply( np.arccos(acos_term) , np.sign(np.dot(np.cross(r_0,r_s),Lhat))) #first term gets us |phi| from 0 to pi, second term gives us the sign
    del acos_term

    phi[phi<0] = phi[phi<0] + 2*pi #make all values range from 0 to pi
    

    #Gmom_L = np.add(Gmom[:,0]*Lhat[0],np.add(Gmom[:,1]*Lhat[1],Gmom[:,2]*Lhat[2]))
    #Gmom_s = np.add(np.multiply(Gmom[:,0],s_hat[:,0]),np.add(np.multiply(Gmom[:,1],s_hat[:,1]),np.multiply(Gmom[:,2],s_hat[:,2])))
    Gmom_r = np.add(np.multiply(Gmom[:,0],r_hat[:,0]),np.add(np.multiply(Gmom[:,1],r_hat[:,1]),np.multiply(Gmom[:,2],r_hat[:,2])))
    
    angMom = np.cross(Gpos, Gmom)
    Lz = np.multiply(angMom[:,0],Lhat_forStats[0]) + np.multiply(angMom[:,1],Lhat_forStats[1]) + np.multiply(angMom[:,2],Lhat_forStats[2])
    #t_hat = np.cross(Lhat,s_hat)
    #Gmom_phi = np.add(np.multiply(Gmom[:,0],t_hat[:,0]),np.add(np.multiply(Gmom[:,1],t_hat[:,1]),np.multiply(Gmom[:,2],t_hat[:,2])))
    Gmom_phi = np.divide(Lz,rmag) 
    del r_z;del r_s;del Lz
    print("Time to convert to disk coordinates:",time.time()-t1)

   

    ##################SORT INTO BINS########################################
    print("Binning everything...")
    xmag = np.multiply(smag,np.cos(phi))
    ymag = np.multiply(smag,np.sin(phi))

    N = np.size(smag)
    
    Gvel_r = np.divide(Gmom_r , Gmas)
    
    sample_cart = np.zeros((N,2))
    sample_cart[:,0]=xmag
    sample_cart[:,1]=ymag;
    #sample_cart[:,2]=Gvel_r

    t1=time.time()

    op = 'sum'
    
    binRange=[[-max_x,max_x],[-max_y,max_y]]

    binned_mass_3d_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,Gmas,op,[nx,ny],range=binRange)
    binned_mom_r_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,Gmom_r,op,[nx,ny],range=binRange)
    binned_mom_phi_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,Gmom_phi,op,[nx,ny],range=binRange)

    #binRange=[[-max_x,max_x],[-max_y,max_y],[-max_vel,max_vel]]

    #binned_mass_3d_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,Gmas,op,[nx,ny,nvel],range=binRange)
    #binned_mom_r_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,Gmom_r,op,[nx,ny,nvel],range=binRange)
    
    if writeStatsFile:
        hf_stats = h5py.File(statsDir,'w')
        hf_stats.create_dataset('pos_center',data = pos_center);
        hf_stats.create_dataset('r0',data=r_0_forStats);
        hf_stats.create_dataset('Lhat',data=Lhat_forStats);
        hf_stats.create_dataset('vel_center',data=vel_center);
        hf_stats.create_dataset('max_x',data=max_x);
        hf_stats.create_dataset('max_y',data=max_y);
        hf_stats.create_dataset('max_vel',data=max_vel);
        hf_stats.create_dataset('max_z',data=max_z);
        #hf_stats.create_dataset('max_s',data=None);
        hf_stats.create_dataset('nx',data=nx);
        hf_stats.create_dataset('ny',data=ny);
        hf_stats.create_dataset('nvel',data=nvel);
        #hf_stats.create_dataset('sbin_size',data=None);
        #hf_stats.create_dataset('phibin_size',data=None);
        hf_stats.create_dataset('ascale',data=ascale);
       # hf_stats.create_dataset('M_stellar',data=None);
        hf_stats.close()



    
    if returnTiltedRingData:
        max_s = max(max_x,max_y)
        max_r_rings=max_s
        ringRes = 0.5
        nRings = np.round(max_r_rings / ringRes).astype(int)
    
        tiltedRing_mass,binedge,binnum = stats.binned_statistic_dd(smag,Gmas  ,op,[nRings],range=[[0,max_r_rings]]) 
    
        tiltedRing_vRad,binedge,binnum = stats.binned_statistic_dd(smag,Gmom_r,op,[nRings],range=[[0,max_r_rings]])
        tiltedRing_vRad = np.divide(tiltedRing_vRad , tiltedRing_mass)
        tiltedRing_vRot,binedge,binnum = stats.binned_statistic_dd(smag,Gmom_phi,op,[nRings],range=[[0,max_r_rings]])
        tiltedRing_vRot = np.divide(tiltedRing_vRot , tiltedRing_mass)
    
        tiltedRing_sigmaGas_x,binedge,binnum = stats.binned_statistic_dd(smag,Gmom[:,0]  ,'std',[nRings],range=[[0,max_r_rings]]) 
        tiltedRing_sigmaGas_y,binedge,binnum = stats.binned_statistic_dd(smag,Gmom[:,1]  ,'std',[nRings],range=[[0,max_r_rings]]) 
        tiltedRing_sigmaGas_z,binedge,binnum = stats.binned_statistic_dd(smag,Gmom[:,2]  ,'std',[nRings],range=[[0,max_r_rings]]) 

        tiltedRing_sigmaGas = np.sqrt(np.power(tiltedRing_sigmaGas_x,2) + np.power(tiltedRing_sigmaGas_y,2) + np.power(tiltedRing_sigmaGas_z,2))
        tiltedRing_sigmaGas = np.divide(tiltedRing_sigmaGas , tiltedRing_mass)
    
        tiltedRing_height,binedge,binnum = stats.binned_statistic_dd(smag , np.multiply(Gmas , np.abs(zmag)),op,[nRings],range=[[0,max_r_rings]]) #not necesarily inclination, but best esitmate I can think of without a fixed sightline?
        tiltedRing_height = np.divide(tiltedRing_height , tiltedRing_mass)
        tiltedRing_inclination = np.arctan2(tiltedRing_height , np.linspace(0,max_r_rings,nRings))
        
        nPhiBins = 360
        #vRad_phiBins,binedge,binnum = stats.binned_statistic_dd(phi,Gmom_r,op,[nPhiBins],range=[[0,2*np.pi]])
        vRad_phiBins,binedge,binnum = stats.binned_statistic_dd([smag,phi],Gmom_r,op,[nRings,nPhiBins],range=[[0,max_r_rings],[0,2*np.pi]])
        
        tiltedRing_sphericalHarmonic = np.zeros((np.shape(tiltedRing_mass)))
        tiltedRing_majorAxisAngle = np.copy(tiltedRing_sphericalHarmonic)
        for i in range(0,np.size(tiltedRing_sphericalHarmonic)):
            vRad_fft = np.fft.fft(vRad_phiBins[i,:])
            tiltedRing_sphericalHarmonic[i] = np.argmax(vRad_fft[0:int(np.size(vRad_fft)/2)]) * nPhiBins/2/np.pi
        #tiltedRing_majorAxisAngle = FitEllipse(xmag,ymag,Gmas)
        smag_phiBins,binedge,binnum = stats.binned_statistic_dd([smag,phi],np.multiply(smag,Gmas),op,[nRings,nPhiBins],range=[[0,max_r_rings],[0,2*np.pi]])
        mass_phiBins,binedge,binnum = stats.binned_statistic_dd([smag,phi],Gmas,op,[nRings,nPhiBins],range=[[0,max_r_rings],[0,2*np.pi]])
        mass_phiBins[mass_phiBins==0]=eps
        smag_phiBins = np.divide(smag_phiBins,mass_phiBins)
        for i in range(0,np.size(tiltedRing_majorAxisAngle)):
            tiltedRing_majorAxisAngle[i] = np.argmax(smag_phiBins[i,:])*2*np.pi/nPhiBins
        
        #tiltedRing_parameters = np.array([np.array(tiltedRing_mass),np.array(tiltedRing_vRad),np.array(tiltedRing_vRot),np.array(tiltedRing_sigmaGas),np.array(tiltedRing_inclination),np.array(tiltedRing_majorAxisAngle),np.array(tiltedRing_sphericalHarmonic)])
        tiltedRing_parameters = np.concatenate((tiltedRing_mass,tiltedRing_vRad,tiltedRing_vRot,tiltedRing_sigmaGas,tiltedRing_inclination,tiltedRing_majorAxisAngle,tiltedRing_sphericalHarmonic))
        
        return binned_mass_3d_cart , binned_mom_r_cart , tiltedRing_parameters

        
    
    #think about way to get inclination, current might be fine. To get harmonic mode, just fit take the fft of the vrad plotted as a funciton of azimuthal angle
    #Only thing left is fitting phi?? plot r as a function of phi and find min?
    

        


    return binned_mass_3d_cart , binned_mom_r_cart, binned_mom_phi_cart




#def FitEllipse(xmag,ymag,,mass):
    #first, find theta angle that minimizes zmag
    #rotate the positions to by this inclination angle and then just fit an ellipse to it
#    pos = np.zeros((np.size(xmag),3))
#    pos[:,0]=xmag
#    pos[:,1]=ymag
#    pos[:,2]=zmag
    
#    for i in np.linspace(-10,10,1000):
#        newPos = 
        


def RunBinfireOnSightline(snapdir,statsDir,sightlineDir,Nsightlines1d,rcDir,Nsnap,output,maxModelRadius,diskRadius,diskHeight,bandwidth_km_s,NspecBins,tempMin=[None],tempMax=[None],densMin=[None],densMax=[None],phasetag=['AG'],rshrinksphere=5000,rminsphere=10,shrinkfactor=0.7):
    Nsnapstring = str(Nsnap)
    shrinking_sphere_flag=0
    if statsDir is not None:
        r_0,pos_center,Lhat,vel_center = ReadStats(statsDir)
    
    sph_shieldLength = None   
    t0 = time.time() #for keeping track of efficiency
    #Resolution Information

    if np.size(np.shape(densMin))==0:
        densMin=[densMin]
    if np.size(np.shape(densMax))==0:
        densMax=[densMax]
    if np.size(np.shape(tempMin))==0:
        tempMin=[tempMin]
    if np.size(np.shape(tempMax))==0:
        tempMax=[tempMax]


    needToCenter = False


    #Read in Header and Data
    header = readsnap_initial(snapdir, Nsnapstring,0,snapshot_name='snapshot',extension='.hdf5',h0=1,cosmological=1, header_only=1)
    ascale = header['time']
    h = header['hubble']

    #Read the Snapshots#################################################
    #Already accounts for factors of h, but not the hubble flow
    Gloaded = False;

    #rTrunc = maxModelRadius
    #truncMax = np.zeros((3))
    #truncMin = np.zeros((3))
    #truncMax[0] = pos_center[0]+rTrunc;truncMax[1] = pos_center[1]+rTrunc;truncMax[2] = pos_center[2]+rTrunc
    #truncMin[0] = pos_center[0]-rTrunc;truncMin[1] = pos_center[1]-rTrunc;truncMin[2] = pos_center[2]-rTrunc

    Gloaded=False
    Sloaded=False
    DMloaded=False


    
    G = readsnap_initial(snapdir, Nsnapstring, 0, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Gas, only load position and density
    Gpos = G['p'] #positions
    Gpos-=pos_center
    r2 = np.add(np.power(Gpos[:,0],2),np.add(np.power(Gpos[:,1],2),np.power(Gpos[:,2],2))) #find distance of particles from center
    truncMask = ((r2 > -1))
    #Gpos=Gpos[truncMask]

    #Gdens = G['rho'] #Densities for finding center

    #if buildShieldLengths:
        #    sph_shieldLength = FindShieldLength(3.086e21*Gpos,404.3*Gdens*0.76) # output units: cm.
        #    sph_shieldLength /= unit_L #convert to sim units


    #sightlineMask,impact,distance,pos_observer,vel_observer,sightline = ReadSightlineFile(sightlineDir)

    t1 = time.time()
    #truncMask =  (Gpos[:,0]<truncMax[0]) & (Gpos[:,0] > truncMin[0]) & (Gpos[:,1] < truncMax[1]) & (Gpos[:,1] > truncMin[1]) & (Gpos[:,2] < truncMax[2]) & (Gpos[:,2] > truncMin[2]) 
    #Gpos = Gpos[sightlineMask]
   # Gdens = Gdens[sightlineMask]
    #if buildShieldLengths:
    #    sph_shieldLength = sph_shieldLength[truncMask]

    G = readsnap_trunc(snapdir, Nsnapstring, 0, truncMask, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Gas, only load truncated data
    Gvel = G['v']#Velocities
    Gmas = G['m'] #masses
    Gz = G['z'] #metallicities
    Gnh = G['nh'] #neutral hydrogen
    Gh = G['h'] #Kernal Lengths for col dens approx
    Gtemp = calcTemps(G['u'],G['ne'],Gz)
    Gloaded = True
    del G
    print("Time to load gas:",time.time()-t1)
    

    #Gpos-=pos_center #Shift to Galactic Frame
    #r2 = np.add(np.power(Gpos[:,0],2),np.add(np.power(Gpos[:,1],2),np.power(Gpos[:,2],2))) #find distance of particles from center
    Gvel-=vel_center
    
    


#CALCULATE MOMENTUMS################################################################################
    N = np.size(Gmas)
    Gmom = np.zeros((N,3));
    Gmom[:,0] = np.multiply(Gmas,Gvel[:,0]) #Calculate each Momentum component
    Gmom[:,1] = np.multiply(Gmas,Gvel[:,1])
    Gmom[:,2] = np.multiply(Gmas,Gvel[:,2])
    
   # del Gvel #Delete for memory, work only in momentum from here out

    #nh = findDensities(Gz,Gdens) #Calculate Density

    t1=time.time()

    r_z = np.zeros((N,3))
    r_s = np.zeros((N,3))

    smag = np.zeros((N))
    zmag = np.zeros((N))

    print("Finding velocity components...")

    zmag = np.dot(Gpos,Lhat)
    r_z[:,0] = zmag*Lhat[0]
    r_z[:,1] = zmag*Lhat[1]
    r_z[:,2] = zmag*Lhat[2]

    r_s = np.subtract(Gpos,r_z)
    smag = VectorArrayMag(r_s)
    smag[smag==0] = eps #make zero entries epsilon for division purposes


    N=np.size(smag) #Redefine number of particles we are looking at

    s_hat = np.zeros((N,3))
    r_hat = np.zeros((N,3))

    s_hat[:,0] = np.divide(r_s[:,0],smag)
    s_hat[:,1] = np.divide(r_s[:,1],smag)
    s_hat[:,2] = np.divide(r_s[:,2],smag)
    
    rmag = VectorArrayMag(Gpos)
    r_hat[:,0] = np.divide(Gpos[:,0],rmag)
    r_hat[:,1] = np.divide(Gpos[:,1],rmag)
    r_hat[:,2] = np.divide(Gpos[:,2],rmag)
    
    t_hat = np.cross(Lhat,s_hat)

    phi = np.zeros((N))
    if r_0 is None:
        print('Defining r0...')
        r_0 = np.zeros((3))
        r_0[:] = r_s[0,:] / np.linalg.norm(r_s[0,:])
    else:
        r_0 = r_0 - np.dot(r_0,Lhat)*Lhat #project onto current disk    
        r_0 = r_0 / np.linalg.norm(r_0) #make unit vector

    print('r0 is',r_0)


    acos_term = np.divide(np.dot(r_s,r_0),(np.linalg.norm(r_0)*smag))
    acos_term[acos_term>1] = 1 #make sure the term isn't above magnitude 1 by a rounding error
    acos_term[acos_term<-1] = -1
    phi = np.multiply( np.arccos(acos_term) , np.sign(np.dot(np.cross(r_0,r_s),Lhat))) #first term gets us |phi| from 0 to pi, second term gives us the sign
    del acos_term

    phi[phi<0] = phi[phi<0] + 2*pi #make all values range from 0 to pi
    del r_z;del r_s   

    Gmom_L = np.add(Gmom[:,0]*Lhat[0],np.add(Gmom[:,1]*Lhat[1],Gmom[:,2]*Lhat[2]))
    Gmom_s = np.add(np.multiply(Gmom[:,0],s_hat[:,0]),np.add(np.multiply(Gmom[:,1],s_hat[:,1]),np.multiply(Gmom[:,2],s_hat[:,2])))
    
    Gmom_r = np.add(np.multiply(Gmom[:,0],r_hat[:,0]),np.add(np.multiply(Gmom[:,1],r_hat[:,1]),np.multiply(Gmom[:,2],r_hat[:,2])))
    
    Gvel_t = np.add(np.multiply(Gmom[:,0],t_hat[:,0]),np.add(np.multiply(Gmom[:,1],t_hat[:,1]),np.multiply(Gmom[:,2],t_hat[:,2])))
    Gvel_t = np.divide(Gvel_t,Gmas) 
    
    hf_RC = h5py.File(rcDir,'r')
    f = interpolate.interp1d(np.array(hf_RC['rPlot']) , np.array(hf_RC['velPhi']))
    hf_RC.close()
    
    dVelPhi = np.zeros((N))
    dVelPhi[rmag<maxModelRadius] = Gvel_t[rmag<maxModelRadius] - f(rmag[rmag<maxModelRadius])
    del Gvel_t


    print("Time to convert to disk coordinates:",time.time()-t1)



    ##################SORT INTO BINS########################################
    print("Binning everything...")
    edgeMat = np.zeros((np.size(smag)))
    edgeMask = ((smag<1.1*diskRadius) & (smag>0.9*diskRadius) & ( np.abs(zmag) < diskHeight ))
    
    print("Smag range for edgeMask=",np.min(smag[edgeMask]),np.max(smag[edgeMask]))
    edgeMat[edgeMask] = 1
    
    binned_x = np.zeros((Nsightlines1d,Nsightlines1d,NspecBins))
    binned_y = np.copy(binned_x)
    binned_z = np.copy(binned_x)
    isEdge = np.copy(binned_x)
    HVC_mass = np.copy(binned_x)
    IVC_mass = np.copy(binned_x)
    binned_mass = np.copy(binned_x)
    binned_mom_r = np.copy(binned_x)

    for i in range(0,Nsightlines1d):
        for j in range(0,Nsightlines1d):
            nSightline = j+i*Nsightlines1d
            sightlineMask,impact,distance,pos_observer,vel_observer,sightline = ReadSightlineFile(sightlineDir+str(nSightline)+'.hdf5')
            gDoppler = GetRadialVelocity(Gpos[sightlineMask]-pos_observer,Gvel[sightlineMask]-vel_observer); #replace Gpos, Spos

            t1=time.time()

            binRange=[[-bandwidth_km_s/2 , bandwidth_km_s/2]]

            binned_mass[i,j,:],binedge,binnum = stats.binned_statistic_dd(gDoppler,Gmas[sightlineMask],'sum',[NspecBins],range=binRange)
            binned_mom_r[i,j,:],binedge,binnum = stats.binned_statistic_dd(gDoppler,Gmom_r[sightlineMask],'sum',[NspecBins],range=binRange)
    
            binned_mass[binned_mass==0]=eps
    
            binned_x[i,j,:],binedge,binnum = stats.binned_statistic_dd(gDoppler,np.multiply(Gmas[sightlineMask] , Gpos[sightlineMask,0]),'sum',[NspecBins],range=binRange)
            binned_x[i,j,:] = np.divide(binned_x[i,j,:],binned_mass[i,j,:])
            binned_y[i,j,:],binedge,binnum = stats.binned_statistic_dd(gDoppler,np.multiply(Gmas[sightlineMask] , Gpos[sightlineMask,1]),'sum',[NspecBins],range=binRange)
            binned_y[i,j,:] = np.divide(binned_y[i,j,:],binned_mass[i,j,:])
            binned_z[i,j,:],binedge,binnum = stats.binned_statistic_dd(gDoppler,np.multiply(Gmas[sightlineMask] , Gpos[sightlineMask,2]),'sum',[NspecBins],range=binRange)
            binned_z[i,j,:] = np.divide(binned_z[i,j,:],binned_mass[i,j,:])
    

            isEdge[i,j,:],binedge,binnum = stats.binned_statistic_dd(gDoppler,edgeMat[sightlineMask],'sum',[NspecBins],range=binRange)
            isEdge[isEdge>0]=1
    
            hvcCutoff=70 #km/s
            ivcCutoff=40
            hvcMask = np.where((dVelPhi[sightlineMask]>hvcCutoff))
            ivcMask = np.where( (dVelPhi[sightlineMask]>ivcCutoff) & (dVelPhi[sightlineMask]<hvcCutoff))
            HVC_mass[i,j,:],binedge,binnum = stats.binned_statistic_dd(gDoppler[hvcMask], Gmas[sightlineMask][hvcMask] , 'sum',[NspecBins] , range=binRange)
            IVC_mass[i,j,:],binedge,binnum = stats.binned_statistic_dd(gDoppler[ivcMask], Gmas[sightlineMask][ivcMask], 'sum', [NspecBins] , range=binRange)
    


    return binned_x, binned_y, binned_z, isEdge, HVC_mass, IVC_mass, binned_mass, binned_mom_r




###################################################################################################################################################################################################

def calcTemps(GintE,ElectronAbundance,Gz): #Calculate Temperatures
    gamma = 1.6666666666666666667
    kb =  1.38064852*10**(-23)*(100.0/unit_L)**2*(1000.0/unit_M)*(unit_T)**2 ## Boltzmann's constant, appropriate units
    #GintE = G['u'] #internal energy
    #ElectronAbundance = G['ne']
    #Gz = G['z']
    N = np.size(GintE)
    
    helium_mass_fraction=np.zeros((N))
    helium_mass_fraction[:] = Gz[:,1]
    y_helium = np.divide(helium_mass_fraction,np.multiply(4,np.subtract(1,helium_mass_fraction)))
    #y_helium = helium_mass_fraction / (4*(1-helium_mass_fraction))
    mu=np.divide(np.add(1,np.multiply(4,y_helium)),np.add(1,np.add(y_helium,ElectronAbundance)))
    #mu = (1 + 4*y_helium) / (1+y_helium+ElectronAbundance)
    mean_molecular_weight = np.multiply(mu,proton_mass)
    #mean_molecular_weight = mu*proton_mass
    Gtemp = np.multiply(mean_molecular_weight,np.multiply(((gamma-1)/kb),GintE))
    #Gtemp = mean_molecular_weight * (gamma-1) * GintE / kb

    return Gtemp


def findCenter(pos_center,Gdens,Gpos,Spos,Smas,old_a,v,ascale,rshrinksphere,rminsphere,shrinkfactor,shrinking_sphere_flag,mask_centers=None,mask_radius=None):

    N = np.size(Gdens)
    NS = np.size(Smas)
    if ((mask_centers is not None) and (mask_radius is not None)): #Mask data to ignore a certain halo/galaxy
        Nmask = np.size(mask_centers)/3 #Number of mask centers
        print("Nmask is",Nmask)
        #Gx = np.zeros((N));Gy = np.copy(Gx);Gz = np.copy(Gx)
        #Sx = np.zeros((NS));Sy = np.copy(Sx);Sz = np.copy(Sx)

        #Gx[:] = Gpos[:,0];Gy[:]=Gpos[:,1];Gz[:] = Gpos[:,2]
        

        for ii in range(0,Nmask): #Mask multiple positions
            if Nmask > 1:
                mask_center = np.zeros((3))
                mask_center[:] = mask_centers[ii,:]
            else:
                mask_center = mask_centers
            print("Masking at: ",mask_center)

            maxima = mask_center+mask_radius;minima = mask_center - mask_radius
            xmax = maxima[0];ymax=maxima[1];zmax=maxima[2]
            xmin = minima[0];ymin=minima[1];zmin=minima[2]
            gasMask = np.where( (Gpos[:,0]<xmax) & (Gpos[:,0] > xmin) & (Gpos[:,1] < ymax) & (Gpos[:,1] > ymin) & (Gpos[:,2] < zmax) & (Gpos[:,2] > zmin) )
            starMask = np.where( (Spos[:,0]<xmax) & (Spos[:,0] > xmin) & (Spos[:,1] < ymax) & (Spos[:,1] > ymin) & (Spos[:,2] < zmax) & (Spos[:,2] > zmin) )
            Gdens[gasMask] = 0 #Won't Modify actual variables
            Smas[starMask] = 0
 
        
        


    if ((old_a is not None) and (v is not None)):
        delta_pos = v*(ascale-old_a)
        pos_center = pos_center+delta_pos 

        
    if pos_center is None:
      tmp_index = np.argmax(Gdens) 
      if not np.isscalar(tmp_index):
          center_index = tmp_index[0]
      else:
          center_index = tmp_index #in case there are degenerate max densities

      pos_center = Gpos[center_index,:] #Naive center of sim.
      print("Center estimate from density: ",pos_center)

    r2_S = np.add(np.power(np.subtract(Spos[:,0],pos_center[0]),2),np.add(np.power(np.subtract(Spos[:,1],pos_center[1]),2),np.power(np.subtract(Spos[:,2],pos_center[2]),2)))
    print("Shape of r2_S",np.shape(r2_S))
    if np.max(r2_S) < rshrinksphere**2:
       rshrinksphere = np.sqrt(np.max(r2_S))

    Sidx = np.where(r2_S<rshrinksphere**2)
    del r2_S #delete for now for memory, will recalculate with optimized com

    if shrinking_sphere_flag==1:
      pos_center = shrink_sphere(Smas[Sidx],Spos[Sidx,:][0],0,rshrinksphere,rminsphere,shrinkfactor)
      print("Best Estimate of Center:",pos_center)
    else:
      print("Skipping Shrinking Sphere...")
      print("Center at:",pos_center)
    return pos_center

def findAngularMomentum(nh,Gpos,Gmom,Gtemp,r2):
    N = np.size(Gtemp)
    L = np.zeros((N,3)) #Initialize Angular Momentum Vector

    print("Finding orientation of disk...")
    L_total = np.zeros((3)) #sum of momentums in center
    L_avg = np.zeros((3)) #average momentum of central disk

    T_cutoff_L = 8000#Kelvin
    dens_cutoff_L = 1*unit_L**3#cm^-3
    r_cutoff_L = 10.0 #kpc for finding the average velocity of the galaxy (averaged over a sphere of radius r_cutoff)
    L = np.cross(Gpos[(Gtemp < T_cutoff_L) & (nh > dens_cutoff_L) & (r2 < r_cutoff_L**2)],Gmom[(Gtemp < T_cutoff_L) & (nh > dens_cutoff_L) & (r2 < r_cutoff_L**2)])
    #L = np.cross(Gpos[(r2 < r_cutoff_L**2)],Gmom[(r2 < r_cutoff_L**2)]) #Look at all phases for whole disk

    L_avg[0] = np.mean(L[:,0])
    L_avg[1] = np.mean(L[:,1])
    L_avg[2] = np.mean(L[:,2])

    L_avg_mag = np.linalg.norm(L_avg)
    Lhat = np.zeros((3))
    Lhat[:] = np.divide(L_avg,L_avg_mag)
    return Lhat



def findDensities(Gz,Gdens):
    N = np.size(Gz[:,0])
    #Find particle densities
    nh=np.zeros((N))
    h_massfrac = np.ones((N))
    h_massfrac = np.subtract(h_massfrac,np.add(Gz[:,0],Gz[:,1])) #1 - mass frac of metals - mass frac of helium
    nh = np.divide(np.multiply(Gdens,h_massfrac),proton_mass)
    return nh


def calcRotationCurve(r2,r2_DM,r2_S,Gmas,DMmas,Smas,max_s,ns):
    rG = np.sqrt(r2)
    rDM = np.sqrt(r2_DM)
    rS = np.sqrt(r2_S)

    Gmas_RC = np.append(Gmas[rG<max_s],[0,0])
    DMmas_RC = np.append(DMmas[rDM<max_s],[0,0])
    Smas_RC = np.append(Smas[rS<max_s],[0,0])
  
    rG_RC = np.append(rG[rG<max_s],[0,max_s])
    rDM_RC = np.append(rDM[rDM<max_s],[0,max_s])
    rS_RC = np.append(rS[rS<max_s],[0,max_s])

    binned_mas_DM,binedge1d,binnum1d = stats.binned_statistic_dd(rDM_RC,DMmas_RC,'sum',ns)
    binned_mas_S,binedge1d,binnum1d = stats.binned_statistic_dd(rS_RC,Smas_RC,'sum',ns)
    binned_mas_G,binedge1d,binnum1d = stats.binned_statistic_dd(rG_RC,Gmas_RC,'sum',ns)

    binned_mas = np.add(binned_mas_DM,np.add(binned_mas_S,binned_mas_G))
    return binned_mas
    



def calcColumnDensity(Gmas,fnh,KernalLengths,density,Gz,sph_shieldLength=None):
    #Inputs = Mass, number density, Kernal Length, density, metallicity
    Z = Gz[:,0] #metal mass (everything not H, He)
    M_H = 1.67353269159582103*np.power(10.0,-27)*(1000.0/unit_M) ##kg to g to unit mass
    mu_H = 2.3*np.power(10.0,-27.0)*(1000.0/unit_M) #'average' mass of hydrogen nucleus from Krumholz&Gnedin 2011

    #sigma_d21 = 1 #dust cross section per H nucleus to 1000 A radiation normalized to MW val
    #R_16 = 1 #rate coefficient for H2 formation on dust grains, normalized to MW val. Dividing these two cancels out metallicity dependence, so I'm keeping as 1

    #chi = 71*(sigma_d21/R_16)*G_o/Gnh #Scaled Radiation Field


    Z_MW = 0.02 #Assuming MW metallicity is ~solar on average (Check!!)
    Z_solar = 0.02 #From Gizmo documentation

    N_ngb = 32.
    #if buildShieldLengths:
    #    sobColDens = np.multiply(sph_shieldLength,density) + np.multiply(KernalLengths,density) / np.power(N_ngb,1./3.) ## kernal Length/(nearest neighbor number)^{1/3{
    #else:
    sobColDens =np.multiply(KernalLengths,density) / np.power(N_ngb,1./3.) #Cheesy approximation of Column density
    tau = np.multiply(sobColDens,Z)*1/(mu_H*Z_MW) * np.power(10.0,-21.0)/(unit_L**(2)) #cm^2 to Unit_L^2
    tau[tau==0]=eps #avoid divide by 0

    chi = 3.1 * (1+3.1*np.power(Z/Z_solar,0.365)) / 4.1 #Approximation

    s = np.divide( np.log(1+0.6*chi+0.01*np.power(chi,2)) , (0.6 *tau) )
    s[s==-4.] = -4.+eps #Avoid divide by zero
    fH2 = np.divide((1 - 0.5*s) , (1+0.25*s)) #Fraction of Molecular Hydrogen from Krumholz & Knedin

    #fH2[fH2>upperApproxLimit] = 1 - 0.75 * s


    lowerApproxLimit = 0.1
    Zprime = np.divide(Z,Z_solar)
    sigma0 = sobColDens * np.power(10.0,4.0) # Surface column density to Msolar/pc^2
    fc = np.divide(tau , 0.066*Zprime*sigma0)# should be~1
    SFR_0 = 0.25*np.power(10.0,-3.0) #Unit_m * kpc^-2 * Gyr -1
   # denom = np.multiply( sobColDens , np.multiply(2.3*np.multiply(fc,Zprime) , np.divide(4.1,1+3.1*np.power(Zprime,0.365))))
    denom = 7.2*tau
    denom[denom==0]=eps
    print("Max fH2 pre approx:",np.min(fH2))
    mask = ((fH2<lowerApproxLimit))
    print("Mean of low fH2 pre approx:",np.mean(fH2[(mask)& (fH2>0)]))
    print(np.size(fH2[fH2<0])," fH2s below zero pre approx")
    fH2[fH2<lowerApproxLimit] = 1.0/3.0 * ( 2.0 - 44.0*2.5*0.066/7.2 * np.divide(chi[fH2<lowerApproxLimit],tau[fH2<lowerApproxLimit]))
    print("Min fH2 post approx:",np.min(fH2))
    print("Max of low fH2 post approx:",np.max(fH2[(fH2<lowerApproxLimit)]))
    print(np.size(fH2[fH2<0])," fH2s below zero post approx")
    fH2[fH2<0] = 0 #Nonphysical negative molecular fractions set to 0
    

    fHe = Gz[:,1]
    fMetals = Gz[:,0]

    
    #NH1=  Mass * Fraction of Hydrogen * Fraction of Hydrogen that is HI / Mass_HI
    NH1 =  Gmas * (1. - fHe - fMetals) * (1. - fH2 - (1 - fnh)) / M_H
    #NH2=  Mass * Fraction of Hydrogen * Fraction of Hydrogen that is H2 / Mass_HI
    NH2 =  Gmas * (1. - fHe - fMetals) * fH2 / M_H #Gives Number of Hydrogen ATOMS in molecules
    #NHion=Mass * Fraction of Hydrogen * Fraction of Ionized Hydrogen / Mass 
    NHion= Gmas * (1. - fHe - fMetals) * (1.-fnh) / M_H
    

    #Nnh = np.multiply(Gmas,( 1 - fH2 - fHion - fHe - fMetals)) / M_H
    #Nih = np.multiply(Gmas,fHion) / M_H #Number of ionized hydrogen
    #fHtotal = (1. - fHe - fMetals )    


    #fHI = np.multiply( np.multiply(Gnh,(1-(Gz[:,0]+Gz[:,1]))) , (1-fH2) ) #Gnh doesn't account for He and metals
    #Nnh =  np.multiply(Gmas,fHI) / M_H #Number of H1 atoms in particles
    return NH1,NH2,NHion


def writeMainFile(output,suffix,binned_mom_s,binned_mom_t,binned_mom_L,binned_mass,binned_angMomMag,binned_angMomZ,binned_mom_s_cart,binned_mom_t_cart,binned_mom_L_cart,binned_mass_cart,binned_dens_cart,binned_angMomMag_cart,binned_angMomZ_cart):
    hf = h5py.File(output+'_main_bins_cyl_'+suffix+'.hdf5','w')
    nonzero_indices = np.where(binned_mass>0)
    binned_mom_s = binned_mom_s[nonzero_indices]
    binned_mom_t = binned_mom_t[nonzero_indices]
    binned_mom_L = binned_mom_L[nonzero_indices]
    binned_mass = binned_mass[nonzero_indices]
    binned_angMomMag = binned_angMomMag[nonzero_indices]
    binned_angMomZ = binned_angMomZ[nonzero_indices]

    hf.create_dataset('mom_s',data=binned_mom_s)
    hf.create_dataset('mom_t',data=binned_mom_t)
    hf.create_dataset('mom_z',data=binned_mom_L)
    hf.create_dataset('mass',data=binned_mass)
    hf.create_dataset('angmom',data=binned_angMomMag)
    hf.create_dataset('angmomZ',data=binned_angMomZ)
    hf.create_dataset('indices',data=nonzero_indices)
    hf.close()


    hf = h5py.File(output+'_main_bins_cart_'+suffix+'.hdf5','w')
    nonzero_indices = np.where(binned_mass_cart>0)
    binned_mom_s_cart = binned_mom_s_cart[nonzero_indices]
    binned_mom_t_cart = binned_mom_t_cart[nonzero_indices]
    binned_mom_L_cart = binned_mom_L_cart[nonzero_indices]
    binned_dens_cart = binned_dens_cart[nonzero_indices]
    binned_mass_cart = binned_mass_cart[nonzero_indices]
    binned_angMomMag = binned_angMomMag_cart[nonzero_indices]
    binned_angMomZ = binned_angMomZ_cart[nonzero_indices]


    hf.create_dataset('mom_s',data=binned_mom_s_cart)
    hf.create_dataset('mom_t',data=binned_mom_t_cart)
    hf.create_dataset('mom_z',data=binned_mom_L_cart)
    hf.create_dataset('mass',data=binned_mass_cart)
    hf.create_dataset('density',data=binned_dens_cart)
    hf.create_dataset('angmom',data=binned_angMomMag)
    hf.create_dataset('angmomZ',data=binned_angMomZ)
    hf.create_dataset('indices',data=nonzero_indices)
    hf.close()


def writeColumnDensity(binned_NH1,binned_NH1_cart,binned_NHion,binned_NHion_cart,output,suffix):
    hf = h5py.File(output+'_H1Dens_bins_cyl_'+suffix+'.hdf5','w')
    hfIon = h5py.File(output+'_HionDens_bins_cyl_'+suffix+'.hdf5','w')

    nonzero_indices = np.where(binned_NH1>0)
    binned_NH1 = binned_NH1[nonzero_indices]
    hf.create_dataset('H1Dens',data=binned_NH1)
    hf.create_dataset('indices',data=nonzero_indices)
    hf.close()

    nonzero_indices = np.where(binned_NHion>0)
    binned_NHion = binned_NHion[nonzero_indices]
    hfIon.create_dataset('HionDens',data=binned_NHion)
    hfIon.create_dataset('indices',data=nonzero_indices)
    hfIon.close()



    hf = h5py.File(output+'_H1Dens_bins_cart_'+suffix+'.hdf5','w')
    hfIon = h5py.File(output+'_HionDens_bins_cart_'+suffix+'.hdf5','w')

    nonzero_indices = np.where(binned_NH1_cart>0)
    binned_NH1_cart = binned_NH1_cart[nonzero_indices]
    hf.create_dataset('H1Dens',data=binned_NH1_cart)
    hf.create_dataset('indices',data=nonzero_indices)
    hf.close()

    nonzero_indices = np.where(binned_NHion_cart>0)
    binned_NHion_cart = binned_NHion_cart[nonzero_indices]
    hfIon.create_dataset('HionDens',data=binned_NHion_cart)
    hfIon.create_dataset('indices',data=nonzero_indices)
    hfIon.close()

def writeMolecularFraction(binned_NH2,binned_NH2_cart,output,suffix):
    hf = h5py.File(output+'_NH2_bins_cyl_'+suffix+'.hdf5','w')
    nonzero_indices = np.where(binned_NH2>0)
    binned_NH2 = binned_NH2[nonzero_indices]
    hf.create_dataset('NH2',data=binned_NH2)
    hf.create_dataset('indices',data=nonzero_indices)
    hf.close()

    hf = h5py.File(output+'_NH2_bins_cart_'+suffix+'.hdf5','w')
    nonzero_indices = np.where(binned_NH2_cart>0)
    binned_NH2_cart = binned_NH2_cart[nonzero_indices]
    hf.create_dataset('NH2',data=binned_NH2_cart)
    hf.create_dataset('indices',data=nonzero_indices)
    hf.close()




def writeRotationCurve(output,suffix,binned_mas):
    enc_mas = np.cumsum(binned_mas)   
    hf = h5py.File(output+'_rotationcurve_'+suffix+'.hdf5','w')
    hf.create_dataset('rotationcurve',data=enc_mas)
    hf.close()

def VectorArrayMag(r):
    r_magnitude = np.sqrt(np.add(np.power(r[:,0],2),np.add(np.power(r[:,1],2),np.power(r[:,2],2))))
    return r_magnitude


def ReadStats(statsDir):
    hf = h5py.File(statsDir,'r')
    posCenter = np.zeros((3))
    velCenter = np.zeros((3))
    Lhat = np.zeros((3))
    r0 = np.zeros((3))
    orientation_maxima = np.zeros((3))

    posCenter[:] = np.array(hf['pos_center'])
    velCenter[:] = np.array(hf['vel_center'])
    Lhat[:] = np.array(hf['Lhat'])
    r0[:] = np.array(hf['r0'])
    orientation_maxima[0] = np.array(hf['max_x'])
    orientation_maxima[1] = np.array(hf['max_y'])
    orientation_maxima[2] = np.array(hf['max_z'])
    
    print("In readstats for:"+statsDir)
    print("PosCenter=",posCenter)

    hf.close()
    return r0,posCenter,Lhat,velCenter


def GenRotationCurve(fileDir,Nsnap,statsDir,rotationCurveDirectory,maxModelRadius,binsize):
    Nsnapstring = str(Nsnap)
    snapdir = fileDir+Nsnapstring
    print("looking for",snapdir)
    r_0,pos_center,Lhat,vel_center = ReadStats(statsDir+Nsnapstring.zfill(4)+".hdf5")
    
    t0 = time.time() #for keeping track of efficiency
    #Resolution Information

    #Read in Header and Data
    header = readsnap_initial(snapdir, Nsnapstring,0,snapshot_name='snapshot',extension='.hdf5',h0=1,cosmological=1, header_only=1)
    ascale = header['time']
    h = header['hubble']

    #Read the Snapshots#################################################
    #Already accounts for factors of h, but not the hubble flow
    Gloaded = False;

    rTrunc = maxModelRadius
    truncMax = np.zeros((3))
    truncMin = np.zeros((3))
    truncMax[0] = pos_center[0]+rTrunc;truncMax[1] = pos_center[1]+rTrunc;truncMax[2] = pos_center[2]+rTrunc
    truncMin[0] = pos_center[0]-rTrunc;truncMin[1] = pos_center[1]-rTrunc;truncMin[2] = pos_center[2]-rTrunc


    G = readsnap_initial(snapdir, Nsnapstring, 0, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Gas, only load position and density
    Gpos = G['p'] #positions
    


    t1 = time.time()
    truncMask =  (Gpos[:,0]<truncMax[0]) & (Gpos[:,0] > truncMin[0]) & (Gpos[:,1] < truncMax[1]) & (Gpos[:,1] > truncMin[1]) & (Gpos[:,2] < truncMax[2]) & (Gpos[:,2] > truncMin[2]) 
    Gpos = Gpos[truncMask]


    G = readsnap_trunc(snapdir, Nsnapstring, 0, truncMask, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Gas, only load truncated data
    Gvel = G['v']#Velocities
    Gmas = G['m'] #masses
    del G
    print("Time to load gas:",time.time()-t1)

    Gpos-=pos_center #Shift to Galactic Frame
    r2 = np.add(np.power(Gpos[:,0],2),np.add(np.power(Gpos[:,1],2),np.power(Gpos[:,2],2))) #find distance of particles from center
    Gvel-=vel_center
    
    N= np.size(Gmas)


#CALCULATE MOMENTUMS################################################################################

    Gmom = np.zeros((N,3));
    Gmom[:,0] = np.multiply(Gmas,Gvel[:,0]) #Calculate each Momentum component
    Gmom[:,1] = np.multiply(Gmas,Gvel[:,1])
    Gmom[:,2] = np.multiply(Gmas,Gvel[:,2])
    
    del Gvel #Delete for memory, work only in momentum from here out

    #nh = findDensities(Gz,Gdens) #Calculate Density

    t1=time.time()

    r_z = np.zeros((N,3))
    r_s = np.zeros((N,3))

    smag = np.zeros((N))
    zmag = np.zeros((N))

    print("Finding velocity components...")

    zmag = np.dot(Gpos,Lhat)
    r_z[:,0] = zmag*Lhat[0]
    r_z[:,1] = zmag*Lhat[1]
    r_z[:,2] = zmag*Lhat[2]

    r_s = np.subtract(Gpos,r_z)
    smag = VectorArrayMag(r_s)
    smag[smag==0] = eps #make zero entries epsilon for division purposes


    N=np.size(smag) #Redefine number of particles we are looking at

    s_hat = np.zeros((N,3))
    r_hat = np.zeros((N,3))

    s_hat[:,0] = np.divide(r_s[:,0],smag)
    s_hat[:,1] = np.divide(r_s[:,1],smag)
    s_hat[:,2] = np.divide(r_s[:,2],smag)
    
    rmag = VectorArrayMag(Gpos)
    r_hat[:,0] = np.divide(Gpos[:,0],rmag)
    r_hat[:,1] = np.divide(Gpos[:,1],rmag)
    r_hat[:,2] = np.divide(Gpos[:,2],rmag)
    
    t_hat = np.cross(Lhat,s_hat)

    phi = np.zeros((N))
    if r_0 is None:
        print('Defining r0...')
        r_0 = np.zeros((3))
        r_0[:] = r_s[0,:] / np.linalg.norm(r_s[0,:])
    else:
        r_0 = r_0 - np.dot(r_0,Lhat)*Lhat #project onto current disk    
        r_0 = r_0 / np.linalg.norm(r_0) #make unit vector

    print('r0 is',r_0)


    acos_term = np.divide(np.dot(r_s,r_0),(np.linalg.norm(r_0)*smag))
    acos_term[acos_term>1] = 1 #make sure the term isn't above magnitude 1 by a rounding error
    acos_term[acos_term<-1] = -1
    phi = np.multiply( np.arccos(acos_term) , np.sign(np.dot(np.cross(r_0,r_s),Lhat))) #first term gets us |phi| from 0 to pi, second term gives us the sign
    del acos_term

    phi[phi<0] = phi[phi<0] + 2*pi #make all values range from 0 to pi
    del r_z;del r_s   
    
    Gmom_t = np.add(np.multiply(Gmom[:,0],t_hat[:,0]),np.add(np.multiply(Gmom[:,1],t_hat[:,1]),np.multiply(Gmom[:,2],t_hat[:,2])))
    
    print("Time to convert to disk coordinates:",time.time()-t1)



    t1=time.time()
    
    binRange=[[0 , maxModelRadius]]
    Nbins = int(4*maxModelRadius)

    binned_mass,binedge,binnum = stats.binned_statistic_dd(rmag,Gmas,'sum',[Nbins],range=binRange)
    binned_mom_r,binedge,binnum = stats.binned_statistic_dd(rmag,Gmom_t,'sum',[Nbins],range=binRange)
    velPhi = np.divide(binned_mom_r,binned_mass)
    rPlot=np.linspace(0,maxModelRadius,Nbins)
    
    hf_RC = h5py.File(rotationCurveDirectory,'w')
    hf_RC.create_dataset('velPhi',data = velPhi)
    hf_RC.create_dataset('rPlot',data = rPlot)
    hf_RC.close()
        



