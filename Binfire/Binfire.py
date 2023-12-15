import numpy as np
import h5py as h5py
import time
import scipy
import scipy.stats as stats
from Binfire.readsnap_binfire import readsnap_initial
from Binfire.readsnap_binfire import readsnap_trunc
from Binfire.readsnap_binfire import ReadStats
from Binfire.CenteringFunctions import FindCenter
from Binfire.CenteringFunctions import FindAngularMomentum
from Binfire.CenteringFunctions import center_of_mass_velocity

unit_M = 10**10 * 1.98855 *10**33 #10^10 solar masses / h !!in grams!! #h accounted for in readsnap
unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
unit_V = 1.0*10.0**5 # 1 km/s !!in cm/s!! Converted by factor of sqrt(a) in readsnap
unit_T = unit_L/unit_V
unit_rho = unit_M / unit_L**3

proton_mass = 1.6726219*10**(-27)*(1000.0/unit_M) ##appropriate units

pi = np.pi
eps = 1e-10

####Functions to read in FIRE snapshots, bin data on a cartesian grid, and return relevant parameters (mass, radial mass flux, and rotational mass flux)
####
####Code initially loads particle data from snapshot. If this snapshot has already been centered, it will read the existing stats file. Otherwise it will
####determine the galactic center, central velocity, and central angular momentum. From here, it will center the gas particles, define a cartesian coordinate system
####with the z-axis oriented on the angular momentum vector of the galaxy, and bin mass, radial mass flux, and azimuhtal mass flux (used for calculating mass weighted rotaional velocities)
####on a cartesian grid. This version of the code is primarily for use with VeryObservableFIRE and CoNNGaFIT
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/14/2023


def RunBinfire(snapdir,statsDir,Nsnap,output,maxima,Nbins,tempMin=[None],tempMax=[None],densMin=[None],densMax=[None],phasetag=['AG'],rshrinksphere=5000,rminsphere=10,shrinkfactor=0.7,inclination=None):
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
    
    t0 = time.time() #for keeping track of efficiency
    #Resolution Information
    max_x = np.pi/2*maxima[0];max_y= np.pi/2*maxima[1];max_vel=maxima[2]/2.
    max_z = np.pi/2*maxima[2]    
    nx,ny,nvel = Nbins

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
    if needToCenter:
        G = readsnap_initial(snapdir, Nsnapstring, 0, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Gas, only load position and density
        Gpos = G['p'] #positions
        Gdens = G['rho'] #Densities for finding center


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



    if not needToCenter:
        G = readsnap_initial(snapdir, Nsnapstring, 0, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Gas, only load position and density
        Gpos = G['p'] #positions
        Gdens = G['rho'] #Densities for finding center


    t1 = time.time()
    truncMask =  (Gpos[:,0]<truncMax[0]) & (Gpos[:,0] > truncMin[0]) & (Gpos[:,1] < truncMax[1]) & (Gpos[:,1] > truncMin[1]) & (Gpos[:,2] < truncMax[2]) & (Gpos[:,2] > truncMin[2]) 
    Gpos = Gpos[truncMask]
    Gdens = Gdens[truncMask]

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

    if needToCenter: ###Load dark matter and star particles only if needed for centering
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
     
        pos_center = FindCenter(pos_center,Gdens,Gpos,Spos,Smas,rshrinksphere,rminsphere,shrinkfactor,shrinking_sphere_flag)
        print("Time to center:",time.time()-t1)

    if Gloaded:
       Gpos-=pos_center #Shift to Galactic Frame
       r2 = np.add(np.power(Gpos[:,0],2),np.add(np.power(Gpos[:,1],2),np.power(Gpos[:,2],2))) #find distance of particles from center
    if DMloaded:
        r2_DM = np.add(np.power(np.subtract(DMpos[:,0],pos_center[0]),2),np.add(np.power(np.subtract(DMpos[:,1],pos_center[1]),2),np.power(np.subtract(DMpos[:,2],pos_center[2]),2)))
        del DMpos
    if Sloaded:
        r2_S = np.add(np.power(np.subtract(Spos[:,0],pos_center[0]),2),np.add(np.power(np.subtract(Spos[:,1],pos_center[1]),2),np.power(np.subtract(Spos[:,2],pos_center[2]),2)))

    if vel_center is None:
        t1=time.time()
        Sidx = (r2_S<15**2) #Find where the stars are within 15kpc of center to calculate velocity of center
        DMidx = (r2_DM<15**2)

        vel_center = center_of_mass_velocity(np.concatenate((Smas[Sidx],DMmas[DMidx]),0),np.concatenate((Svel[Sidx,:],DMvel[DMidx,:]),0))
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
        Lhat = FindAngularMomentum(nh,Gpos,Gmom,Gtemp,r2) #Find orientation of the Disk
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
    
    Gmom_r = np.add(np.multiply(Gmom[:,0],r_hat[:,0]),np.add(np.multiply(Gmom[:,1],r_hat[:,1]),np.multiply(Gmom[:,2],r_hat[:,2])))
    
    angMom = np.cross(Gpos, Gmom)
    Lz = np.multiply(angMom[:,0],Lhat_forStats[0]) + np.multiply(angMom[:,1],Lhat_forStats[1]) + np.multiply(angMom[:,2],Lhat_forStats[2])

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

    t1=time.time()

    op = 'sum'
    
    binRange=[[-max_x,max_x],[-max_y,max_y]]

    binned_mass_3d_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,Gmas,op,[nx,ny],range=binRange)
    binned_mom_r_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,Gmom_r,op,[nx,ny],range=binRange)
    binned_mom_phi_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,Gmom_phi,op,[nx,ny],range=binRange)
    
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
        hf_stats.create_dataset('nx',data=nx);
        hf_stats.create_dataset('ny',data=ny);
        hf_stats.create_dataset('nvel',data=nvel);
        hf_stats.create_dataset('ascale',data=ascale);
        hf_stats.close()   


    return binned_mass_3d_cart , binned_mom_r_cart, binned_mom_phi_cart



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


def findDensities(Gz,Gdens):
    N = np.size(Gz[:,0])
    #Find particle densities
    nh=np.zeros((N))
    h_massfrac = np.ones((N))
    h_massfrac = np.subtract(h_massfrac,np.add(Gz[:,0],Gz[:,1])) #1 - mass frac of metals - mass frac of helium
    nh = np.divide(np.multiply(Gdens,h_massfrac),proton_mass)
    return nh

def VectorArrayMag(r):
    r_magnitude = np.sqrt(np.add(np.power(r[:,0],2),np.add(np.power(r[:,1],2),np.power(r[:,2],2))))
    return r_magnitude