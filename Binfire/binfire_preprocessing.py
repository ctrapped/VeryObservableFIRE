import numpy as np
import h5py as h5py
import os.path
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import math
import sys
import time
import scipy.stats as stats
from readsnap_binfire import *
from shrinking_sphere import *
from build_shieldLengths import FindShieldLength #From Matt Orr


unit_M = 10**10 * 1.98855 *10**33 #10^10 solar masses / h !!in grams!! #h accounted for in readsnap
unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
unit_V = 1.0*10.0**5 # 1 km/s !!in cm/s!! Converted by factor of sqrt(a) in readsnap
unit_T = unit_L/unit_V
unit_rho = unit_M / unit_L**3

proton_mass = 1.6726219*10**(-27)*(1000.0/unit_M) ##appropriate units

pi = 3.14159265359
eps = 0.00000000000000000000000000000000000000000000001

buildShieldLengths=False

def binfire_preprocessing(snapdir,Nsnapstring,output,suffix,maxima,res,tempMin,tempMax,densMin,densMax,phasetag,Nmetals,writeFlags,r_0=None,shrinking_sphere_flag=1,pos_center=None,Lhat=None,vel_center=None,time_array=None,old_a=None,old_v=None,rshrinksphere=5000,rminsphere=10,shrinkfactor=0.7,mask_center=None,mask_radius=None):

    sph_shieldLength = None
    writeMain,writeColDens,writeMolFrac,writeStars,writeRC = writeFlags
   
    t0 = time.time() #for keeping track of efficiency

    #Resolution Information
    max_x = maxima[0];max_y=maxima[1];max_z=maxima[2];max_s=maxima[3]
    xbin_size = res[0];ybin_size=res[1];zbin_size=res[2];sbin_size=res[3];phibin_size=res[4]
    ns = int(math.ceil(max_s/sbin_size))
    nphi = int(np.round(2*pi/phibin_size))
    nz = int(math.ceil(2*max_z/zbin_size))
    nx = int(math.ceil(2*max_x/xbin_size))
    ny = int(math.ceil(2*max_y/ybin_size))



    if np.size(np.shape(densMin))==0:
        densMin=[densMin]
    if np.size(np.shape(densMax))==0:
        densMax=[densMax]
    if np.size(np.shape(tempMin))==0:
        tempMin=[tempMin]
    if np.size(np.shape(tempMax))==0:
        tempMax=[tempMax]

    #if densMin is not None:
    #    Nphases = np.size(densMin)
    #    for pp in range(0,Nphases):
    #        if densMin[pp] is not None:
    #            densMin[pp] = densMin[pp]*unit_L**3
    #if densMax is not None:
    #    Nphases = np.size(densMin)
    #    for pp in range(0,Nphases):
    #        if densMax[pp] is not None:
    #            densMax[pp] = densMax[pp]*unit_L**3 #Convert to simulation coordinates

    needToCenter = False
    if ((shrinking_sphere_flag) or (pos_center is None) or (vel_center is None)):
        needToCenter = True


    #Read in Header and Data
    header = readsnap_initial(snapdir, Nsnapstring,0,snapshot_name='snapshot',extension='.hdf5',h0=1,cosmological=1, header_only=1)
    ascale = header['time']
    h = header['hubble']
    time_idx = np.where(time_array == ascale) #Find where in the stored scale factor values this snapshot is
    if np.size(time_idx)>1:
        time_idx = time_idx[0][0]
        print(time_idx)
        print("Warning, degenerate scale factor labels...")
    if np.size(np.shape(time_idx))>0:
        time_idx=time_idx[0]
    print("Time_idx is:",time_idx)

    #Read the Snapshots#################################################
    #Already accounts for factors of h, but not the hubble flow
    Gloaded = False;DMloaded=False;Sloaded=False

    if (needToCenter):
        G = readsnap_initial(snapdir, Nsnapstring, 0, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Gas, only load position and density
        Gpos = G['p'] #positions
        Gdens = G['rho'] #Densities for finding center


        if buildShieldLengths:
            t_shield = time.time()
            sph_shieldLength = FindShieldLength(3.086e21*Gpos,404.3*Gdens*0.76) # output units: cm.
            sph_shieldLength /= unit_L #convert to sim units
            print("Time to calculate shield lengths:", time.time()-t_shield)

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
        rTrunc = max(max_s,max_z)
        rTrunc = max(rTrunc,max_x)
    truncMax = np.zeros((3))
    truncMin = np.zeros((3))
    truncMax[0] = pos_center[0]+rTrunc;truncMax[1] = pos_center[1]+rTrunc;truncMax[2] = pos_center[2]+rTrunc
    truncMin[0] = pos_center[0]-rTrunc;truncMin[1] = pos_center[1]-rTrunc;truncMin[2] = pos_center[2]-rTrunc

    if ((needToCenter) or (writeMain) or (writeRC) or (writeColDens) or (writeMolFrac)):
        if not needToCenter:
            G = readsnap_initial(snapdir, Nsnapstring, 0, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Gas, only load position and density
            Gpos = G['p'] #positions
            Gdens = G['rho'] #Densities for finding center

            if buildShieldLengths:
                sph_shieldLength = FindShieldLength(3.086e21*Gpos,404.3*Gdens*0.76) # output units: cm.
                sph_shieldLength /= unit_L #convert to sim units

        t1 = time.time()
        truncMask =  (Gpos[:,0]<truncMax[0]) & (Gpos[:,0] > truncMin[0]) & (Gpos[:,1] < truncMax[1]) & (Gpos[:,1] > truncMin[1]) & (Gpos[:,2] < truncMax[2]) & (Gpos[:,2] > truncMin[2]) 
        Gpos = Gpos[truncMask]
        Gdens = Gdens[truncMask]
        if buildShieldLengths:
            sph_shieldLength = sph_shieldLength[truncMask]

        G = readsnap_trunc(snapdir, Nsnapstring, 0, truncMask, Nmetals=Nmetals, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Gas, only load truncated data
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

    if (writeRC):
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


    if ((needToCenter) or (writeStars) or (writeRC) or (writeMain)):
        t1=time.time()
        S = readsnap_initial(snapdir, Nsnapstring, 4, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Stars, only load position
        Spos = S['p']
        truncMask =  ((Spos[:,0]<truncMax[0]) & (Spos[:,0] > truncMin[0]) & (Spos[:,1] < truncMax[1]) & (Spos[:,1] > truncMin[1]) & (Spos[:,2] < truncMax[2]) & (Spos[:,2] > truncMin[2])) 
        Spos = Spos[truncMask]

        S = readsnap_trunc(snapdir, Nsnapstring, 4, truncMask, snapshot_name='snapshot', extension='.hdf5',h0=1,cosmological=1) #Stars Matter, only load truncated data
        if (needToCenter): #only need stellar velocities to determine galactic vel
            Svel = S['v']
        Smas = S['m']
        Sage = S['age']
        print("Sage size:",np.size(Sage))
        #if np.max(Sage)>1:
            #print("Warning!! Maxima of Sage is:",np.max(Sage)
        NSt = np.size(Smas) #Number of star particles
        del S
        del truncMask
        Sloaded = True
        print("Time to load stars:",time.time()-t1)


    if needToCenter:
        t1=time.time()
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
        total_stellar_mass = np.sum(Smas[r2_S < 100**2]) #Total mass of stars in 100kpc, for determining stellar half mass radius.    

    if (vel_center is None):
        t1=time.time()
        Sidx = (r2_S<15**2) #Find where the stars are within 15kpc of center to calculate velocity of center
        DMidx = (r2_DM<15**2)
        #Gidx = (r2<10**2)
        #vel_center = center_of_mass_velocity(Smas[Sidx],Svel[Sidx,0][0],Svel[Sidx,1][0],Svel[Sidx,2][0])
        print("Vel center was:",vel_center)

        #vel_center = center_of_mass_vel_gen(Smas[Sidx],Svel[Sidx,:])
        #vel_center = center_of_mass_vel_gen(DMmas[DMidx],DMvel[DMidx,:])
        #vel_center = center_of_mass_vel_gen(Gmas[Gidx],Gvel[Gidx,:])
        vel_center = center_of_mass_vel_gen(np.concatenate((Smas[Sidx],DMmas[DMidx]),0),np.concatenate((Svel[Sidx,:],DMvel[DMidx,:]),0))
        print("Time to find vel_center:",time.time()-t1)

    print("Best Estimate of Center Velocity:")
    print(vel_center)

    if Gloaded:
       Gvel-=vel_center

    if Sloaded:
        Spos-=pos_center

    if needToCenter:
        del Svel #Don't need star velocities anymore


    if (writeRC):
        t1=time.time()
        binned_mas = calcRotationCurve(r2,r2_DM,r2_S,Gmas,DMmas,Smas,max_s,ns)
        print("Time to calculate Rotation Curve:",time.time()-t1)
        t1=time.time()
        writeRotationCurve(output,suffix,binned_mas)
        print("Time to write rotation curve:",time.time()-t1)



#CALCULATE MOMENTUMS################################################################################

    if ((writeMain) or (writeColDens) or (writeMolFrac)):
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
        smag = vectorArrayMag(r_s)
        smag[smag==0] = eps #make zero entries epsilon for division purposes

########Apply location cuts here#################################################
        mask=((smag<max_s) & (abs(zmag)<max_z))
        Gmom=Gmom[mask]
        Gtemp = Gtemp[mask]
        nh=nh[mask]
        r_z=r_z[mask]
        r_s=r_s[mask]
        smag=smag[mask]
        zmag=zmag[mask]
        Gmas=Gmas[mask]
        Gdens=Gdens[mask]
        Gpos=Gpos[mask]
        Gnh = Gnh[mask]
        Gh = Gh[mask]
        Gz = Gz[mask]
        if buildShieldLengths:
            sph_shieldLength = sph_shieldLength[mask]
        N=np.size(smag) #Redefine number of particles we are looking at
        del mask  
##################################################################################




        Gmom_s = np.zeros((N))
        Gmom_L = np.zeros((N))
        Gmom_t = np.zeros((N))
        s_hat = np.zeros((N,3))
        t_hat = np.zeros((N,3))

        s_hat[:,0] = np.divide(r_s[:,0],smag)
        s_hat[:,1] = np.divide(r_s[:,1],smag)
        s_hat[:,2] = np.divide(r_s[:,2],smag)
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
    

        Gmom_L[:] = np.add(Gmom[:,0]*Lhat[0],np.add(Gmom[:,1]*Lhat[1],Gmom[:,2]*Lhat[2]))
        Gmom_t[:] = np.add(np.multiply(Gmom[:,0],t_hat[:,0]),np.add(np.multiply(Gmom[:,1],t_hat[:,1]),np.multiply(Gmom[:,2],t_hat[:,2])))
        Gmom_s[:] = np.add(np.multiply(Gmom[:,0],s_hat[:,0]),np.add(np.multiply(Gmom[:,1],s_hat[:,1]),np.multiply(Gmom[:,2],s_hat[:,2])))

        del r_z;del r_s   
        print("Time to convert to disk coordinates:",time.time()-t1)


        #Apply Phase Cuts
        t1=time.time()
        if tempMax is not None:
            Nphases = np.size(tempMax)
        elif tempMin is not None:
            Nphases = np.size(tempMin)
        elif densMax is not None:
            Nphases = np.size(densMax)
        elif densMin is not None:
            Nphases = np.size(densMin)
        
        if Nphases>1: #Only waste memory if necesarry. Can run 1 phase at a time if memory is an issue
            Gmas0 = np.copy(Gmas)
            Gpos0 = np.copy(Gpos)
            Gdens0= np.copy(Gdens)
            Gmom0 = np.copy(Gmom)
            nh0 = np.copy(nh)
            Gnh0 = np.copy(Gnh)
            Gh0 = np.copy(Gh)
            Gz0 = np.copy(Gz)
            Gmom_s0 = np.copy(Gmom_s)
            Gmom_t0 = np.copy(Gmom_t)
            Gmom_L0 = np.copy(Gmom_L)
            phi0 = np.copy(phi)
            smag0 = np.copy(smag)
            zmag0 = np.copy(zmag)
        

        for pp in range(0,Nphases):
            print("Running phase",pp,":",tempMin[pp],"<T<",tempMax[pp],"and",densMin[pp],"<nh<",densMax[pp])
            if pp>0: #Go back to full vector for subsequent phase cuts.
                t2=time.time()
                Gmas = np.copy(Gmas0)
                Gpos = np.copy(Gpos0)
                Gdens= np.copy(Gdens0)
                Gmom = np.copy(Gmom0)
                nh = np.copy(nh0)
                Gnh = np.copy(Gnh0)
                Gh = np.copy(Gh0)
                Gz = np.copy(Gz0)
                Gmom_s = np.copy(Gmom_s0)
                Gmom_t = np.copy(Gmom_t0)
                Gmom_L = np.copy(Gmom_L0)
                phi = np.copy(phi0)
                smag = np.copy(smag0)
                zmag = np.copy(zmag0)
                print("Time to copy phasecut data",time.time()-t2)
       
            if (tempMax[pp] is not None):
                mask = (Gtemp < tempMax[pp])
                Gmas = Gmas[mask]
                Gpos = Gpos[mask]
                Gdens = Gdens[mask]
                Gmom = Gmom[mask]
                nh = nh[mask]
                Gtemp = Gtemp[mask]
                Gnh = Gnh[mask]
                Gh = Gh[mask]
                Gz = Gz[mask]
                Gmom_s=Gmom_s[mask]
                Gmom_t=Gmom_t[mask]
                Gmom_L=Gmom_L[mask]
                phi=phi[mask]
                smag=smag[mask]
                zmag=zmag[mask]
                if buildShieldLengths:
                    sph_shieldLength = sph_shieldLength[mask]
                del mask
            if (tempMin[pp] is not None):
                mask = (Gtemp > tempMin[pp])
                Gmas = Gmas[mask]
                Gpos = Gpos[mask]
                Gdens = Gdens[mask]
                Gmom = Gmom[mask]
                nh = nh[mask]
                Gtemp = Gtemp[mask]
                Gnh = Gnh[mask]
                Gh = Gh[mask]
                Gz = Gz[mask]
                Gmom_s=Gmom_s[mask]
                Gmom_t=Gmom_t[mask]
                Gmom_L=Gmom_L[mask]
                phi=phi[mask]
                smag=smag[mask]
                zmag=zmag[mask]
                if buildShieldLengths:
                    sph_shieldLength = sph_shieldLength[mask]
                del mask
            if (densMax[pp] is not None):
                mask = (nh < densMax[pp])
                Gmas = Gmas[mask]
                Gpos = Gpos[mask]
                Gdens = Gdens[mask]
                Gmom = Gmom[mask]
                nh = nh[mask]
                Gtemp = Gtemp[mask]
                Gnh = Gnh[mask]
                Gh = Gh[mask]
                Gz = Gz[mask]
                Gmom_s=Gmom_s[mask]
                Gmom_t=Gmom_t[mask]
                Gmom_L=Gmom_L[mask]
                phi=phi[mask]
                smag=smag[mask]
                zmag=zmag[mask]
                sph_shieldLength = sph_shieldLength[mask]
                del mask
            if (densMin[pp] is not None):
                mask = (nh > densMin[pp])
                Gmas = Gmas[mask]
                Gpos = Gpos[mask]
                Gdens = Gdens[mask]
                Gmom = Gmom[mask]
                nh = nh[mask]
                Gtemp = Gtemp[mask]
                Gnh = Gnh[mask]
                Gh = Gh[mask]
                Gz = Gz[mask]
                Gmom_s=Gmom_s[mask]
                Gmom_t=Gmom_t[mask]
                Gmom_L=Gmom_L[mask]
                phi=phi[mask]
                smag=smag[mask]
                zmag=zmag[mask]
                if buildShieldLengths:
                    sph_shieldLength = sph_shieldLength[mask]
                del mask
            print("Time to apply phase cuts:",time.time()-t1)

            NH1,NH2,NHion = calcColumnDensity(Gmas,Gnh,Gh,Gdens,Gz,sph_shieldLength)
            del sph_shieldLength

        ##################SORT INTO BINS########################################
            print("Binning everything...")

            total_sbins = int(math.ceil(max_s/sbin_size))
            total_zbins = int(math.ceil(2*max_z/zbin_size))
            total_phibins = int(np.round(2*pi/phibin_size))
 

            #Trick to bin it from the appropriate ranges by adding 0 mass particles
        
            Gmom_s = np.append(Gmom_s,[0.0,0.0])
            Gmom_t = np.append(Gmom_t,[0.0,0.0])
            Gmom_L = np.append(Gmom_L,[0.0,0.0])

            Gmas = np.append(Gmas,[0.0,0.0])
            Gdens = np.append(Gdens,[0.0,0.0]) 
            NH1 = np.append(NH1,[0.0,0.0])
            NH2 = np.append(NH2,[0.0,0.0])
            NHion = np.append(NHion,[0.0,0.0])

            #Gmom = Gmom[((smag<max_s) & (abs(zmag)<max_z))]
            #Gpos = Gpos[((smag<max_s) & (abs(zmag)<max_z))]

            phi = np.append(phi,[0,2*pi])
            smag = np.append(smag,[0.0,max_s])
            zmag = np.append(zmag,[-max_z,max_z])

            xmag = np.multiply(smag,np.cos(phi))
            ymag = np.multiply(smag,np.sin(phi))
            xmag[np.size(xmag)-1] = max_x;xmag[np.size(xmag)-2] = -max_x #Ensure the cartesian bins are appropriately spaced as well
            ymag[np.size(ymag)-1] = max_y;ymag[np.size(ymag)-2] = -max_y

            N = np.size(smag)
            angMom = np.cross(Gpos,Gmom)
            angMomMag = np.zeros((N))
            angMomMag[:] = np.append(vectorArrayMag(angMom),[0.0,0.0])
            angMomZ = np.zeros((N))
            angMomZ[:] = np.append(angMom[:,0]*Lhat[0]+angMom[:,1]*Lhat[1]+angMom[:,2]*Lhat[2],[0.0,0.0]) 
            del angMom
        
            sample = np.zeros((N,3))
            sample[:,0]=smag;sample[:,1]=phi;sample[:,2]=zmag

            sample_cart = np.zeros((N,3))
            sample_cart[:,0]=xmag;sample_cart[:,1]=ymag;sample_cart[:,2]=zmag

            if (writeMain):
                t1=time.time()

                op = 'std'
                #op = 'sum'
                binned_mass_3d,binedge,binnum = stats.binned_statistic_dd(sample,Gmas,op,[ns,nphi,nz])
                binned_mom_s,binedge,binnum = stats.binned_statistic_dd(sample,Gmom_s,op,[ns,nphi,nz])
                binned_mom_t,binedge,binnum = stats.binned_statistic_dd(sample,Gmom_t,op,[ns,nphi,nz])
                binned_mom_L,binedge,binnum = stats.binned_statistic_dd(sample,Gmom_L,op,[ns,nphi,nz])
                binned_angMomMag,binedge,binnum = stats.binned_statistic_dd(sample,angMomMag,op,[ns,nphi,nz])
                binned_angMomZ,binedge,binnum = stats.binned_statistic_dd(sample,angMomZ,op,[ns,nphi,nz])

                binned_mass_3d_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,Gmas,op,[nx,ny,nz])
                binned_dens_3d_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,Gdens,op,[nx,ny,nz])
                binned_mom_s_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,Gmom_s,op,[nx,ny,nz])
                binned_mom_t_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,Gmom_t,op,[nx,ny,nz])
                binned_mom_L_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,Gmom_L,op,[nx,ny,nz])
                binned_angMomMag_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,angMomMag,op,[nx,ny,nz])
                binned_angMomZ_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,angMomZ,op,[nx,ny,nz])

                print("Time to bin main:",time.time()-t1)
                t1=time.time()
                writeMainFile(output+phasetag[pp],suffix,binned_mom_s,binned_mom_t,binned_mom_L,binned_mass_3d,binned_angMomMag,binned_angMomZ,binned_mom_s_cart,binned_mom_t_cart,binned_mom_L_cart,binned_mass_3d_cart,binned_dens_3d_cart,binned_angMomMag_cart,binned_angMomZ_cart)
                print("Time to write main file:",time.time()-t1)
    

            if (writeColDens or writeMolFrac):
                t1=time.time()

                if writeColDens:
                    t1=time.time()
                    binned_NH1,binedge,binnum = stats.binned_statistic_dd(sample,NH1,'sum',[ns,nphi,nz])
                    binned_NH1_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,NH1,'sum',[nx,ny,nz])
                    binned_NHion,binedge,binnum = stats.binned_statistic_dd(sample,NHion,'sum',[ns,nphi,nz])
                    binned_NHion_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,NHion,'sum',[nx,ny,nz])
                    print("Time to bin column density:",time.time()-t1)
                    t1=time.time()
                    writeColumnDensity(binned_NH1,binned_NH1_cart,binned_NHion,binned_NHion_cart,output+phasetag[pp],suffix) 
                    print("Time to write column density:",time.time()-t1)
                if writeMolFrac:
                    t1=time.time()
                    binned_NH2,binedge,binnum = stats.binned_statistic_dd(sample,NH2,'sum',[ns,nphi,nz])
                    binned_NH2_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,NH2,'sum',[nx,ny,nz])
                    print("Time to bin molecular fraction:",time.time()-t1)
                    t1=time.time()
                    writeMolecularFraction(binned_NH2,binned_NH2_cart,output+phasetag[pp],suffix)
                    print("Time to write molecular fraction:",time.time()-t1)


        fid_stats = open(output+'_stats_'+suffix+'.txt','w')
        fid_stats.write("max_x(0)    max_y(1)    max_z(2)    max_s(3)    xbin_size(4)    ybin_size(5)    zbin_size(6)    sbin_size(7)    phibin_size(8)    x_center(9)    y_center(10)    z_center(11)    ascale(12)    r0_x(13)    r0_y(14)    r0_z(15)    M_stellar(16) Lhat_x(17)    Lhat_y(18)    Lhat_z(19)    vx_center(20)    vy_center(21)    vz_center(22)\n")


        fid_stats.write("%.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f    %.17f\n" % (max_x,max_y,max_z,max_s,xbin_size,ybin_size,zbin_size,sbin_size,phibin_size,pos_center[0],pos_center[1],pos_center[2],ascale,r_0[0],r_0[1],r_0[2],total_stellar_mass,Lhat[0],Lhat[1],Lhat[2],vel_center[0],vel_center[1],vel_center[2]))
        fid_stats.close()


    if writeStars:
        t1=time.time()
        NSt = np.size(Smas[(r2_S < max_s**2)]) #Number of star particles in region
        r_s_stars = np.zeros((NSt,3))
        r_z_stars = np.zeros((NSt,3))


        Spos = Spos[(r2_S < max_s**2)] #Only look at stars within the sphere of interest
        Sage = Sage[(r2_S < max_s**2)]
        Smas = Smas[(r2_S < max_s**2)]

        zmag_stars = np.dot(Spos,Lhat)

        r_z_stars[:,0] = zmag_stars*Lhat[0]
        r_z_stars[:,1] = zmag_stars*Lhat[1]
        r_z_stars[:,2] = zmag_stars*Lhat[2]

        r_s_stars = np.subtract(Spos,r_z_stars)
        del r_z_stars
        smag_stars = np.sqrt(np.add(np.power(r_s_stars[:,0],2),np.add(np.power(r_s_stars[:,1],2),np.power(r_s_stars[:,2],2))))
        smag_stars[smag_stars==0]=eps #make zero entries epsilon for division purposes


        if r_0 is None:
            print('Defining r0...')
            r_0 = np.zeros((3))
            r_0[:] = r_s[0,:] / np.linalg.norm(r_s[0,:])
        else:
            r_0 = r_0 - np.dot(r_0,Lhat)*Lhat #project onto current disk    
            r_0 = r_0 / np.linalg.norm(r_0) #make unit vector

        acos_term = np.divide(np.dot(r_s_stars,r_0),(np.linalg.norm(r_0)*smag_stars))
        acos_term[acos_term>1] = 1 #make sure the term isn't above magnitude 1 by a rounding error
        acos_term[acos_term<-1] = -1
        phi_stars = np.multiply( np.arccos(acos_term) , np.sign(np.dot(np.cross(r_0,r_s_stars),Lhat))) #first term gets us |phi| from 0 to pi, second term gives us the sign
        phi_stars[phi_stars<0] = phi_stars[phi_stars<0] + 2*pi #make all values range from 0 to pi
        del r_s_stars

        xmag_stars = np.multiply(smag_stars,np.cos(phi_stars))
        ymag_stars = np.multiply(smag_stars,np.sin(phi_stars))

        smag_stars = np.append(smag_stars,[0.0,max_s]) #trick to get binned stat to sample symmetrically
        phi_stars = np.append(phi_stars,[0.0,2*pi])
        xmag_stars = np.append(xmag_stars,[-max_x,max_x])
        ymag_stars = np.append(ymag_stars,[-max_y,max_y])
        zmag_stars = np.append(zmag_stars,[-max_z,max_z])

        Smas = np.append(Smas,[0,0])
        Sage = np.append(Sage,[0,0])
        print("Time to convert stars to disk coordinates:",time.time()-t1)
        t1=time.time()
        print("Binning...")

        sample = np.zeros((NSt+2,3))
        sample[:,0]=smag_stars;sample[:,1]=phi_stars;sample[:,2]=zmag_stars

        sample_cart = np.zeros((NSt+2,3))
        sample_cart[:,0]=xmag_stars;sample_cart[:,1]=ymag_stars;sample_cart[:,2]=zmag_stars
        
        ns = int(math.ceil(max_s/sbin_size))
        nphi = int(np.round(2*pi/phibin_size))
        nz = int(math.ceil(2*max_z/zbin_size))

        nx = int(math.ceil(2*max_x/xbin_size))
        ny = int(math.ceil(2*max_y/ybin_size)) 

        binned_mass,binedge,binnum = stats.binned_statistic_dd(sample,Smas,'sum',[ns,nphi,nz])
        binned_mass_cart,binedge,binnum = stats.binned_statistic_dd(sample_cart,Smas,'sum',[nx,ny,nz])
        print("Time to bin stars:",time.time()-t1)
        t1=time.time()
 
        age_idx = np.where(Sage > time_array[time_idx+1]) #Older than the next snapshot (going backwards)
        Smas = Smas[age_idx]#Apply Age cut
        N = np.size(Smas)
        if N>0:
            sample = np.zeros((N+2,3))
            sample[:,0]=np.append(smag_stars[age_idx],[0.0,max_s]) #Again, make sure binning centers correctly
            sample[:,1]= np.append(phi_stars[age_idx],[0.0,2*pi])
            sample[:,2]=np.append(zmag_stars[age_idx],[-max_z,max_z])

            sample_cart = np.zeros((N+2,3))
            sample_cart[:,0]=np.append(xmag_stars[age_idx],[-max_x,max_x])
            sample_cart[:,1]=np.append(ymag_stars[age_idx],[-max_y,max_y])
            sample_cart[:,2]=np.append(zmag_stars[age_idx],[-max_z,max_z])
            
            binned_mass_new,binedge,binnum = stats.binned_statistic_dd(sample,np.append(Smas,[0.0,0.0]),'sum',[ns,nphi,nz])
            binned_mass_cart_new,binedge,binnum = stats.binned_statistic_dd(sample_cart,np.append(Smas,[0.0,0.0]),'sum',[nx,ny,nz]) #Only grab the stars formed since the last snapshot
        else:
            binned_mass_new = np.zeros((ns,nphi,nz))
            binned_mass_cart_new = np.zeros((nx,ny,nz))
        print("Time to bin stellar age cuts:",time.time()-t1)
        t1=time.time()

        hf = h5py.File(output+'_stars_bin_cyl_'+suffix+'.hdf5','w')
        nonzero_indices = np.where(binned_mass>0)
        binned_mass = binned_mass[nonzero_indices]
        binned_mass_new = binned_mass_new[nonzero_indices]

        hf.create_dataset('mass',data=binned_mass)
        hf.create_dataset('newMass',data=binned_mass_new)
        hf.create_dataset('indices',data=nonzero_indices)
        hf.close() 


        hf = h5py.File(output+'_stars_bin_cart_'+suffix+'.hdf5','w')
        nonzero_indices = np.where(binned_mass_cart>0)
        binned_mass = binned_mass_cart[nonzero_indices]
        binned_mass_new = binned_mass_cart_new[nonzero_indices]

        hf.create_dataset('mass',data=binned_mass)
        hf.create_dataset('newMass',data=binned_mass_new)
        hf.create_dataset('indices',data=nonzero_indices)
        hf.close()
        print("Time to write stellar data:",time.time()-t1)       
        


    print("Done!")
    t1 = time.time()
    print("Total Time: ",t1-t0)
    return r_0,pos_center,ascale


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

def vectorArrayMag(r):
    r_magnitude = np.sqrt(np.add(np.power(r[:,0],2),np.add(np.power(r[:,1],2),np.power(r[:,2],2))))
    return r_magnitude






