import numpy as np
import h5py as h5py
import os.path

####Legacy io functions to read in FIRE snapshots
####
####Modified By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/14/2023

def readsnap_initial(sdir,snum,ptype,
    snapshot_name='snapshot',
    extension='.hdf5',
    h0=0,cosmological=0,skip_bh=0,four_char=0,
    header_only=0,loud=0):
    
    if (ptype<0): return {'k':-1};
    if (ptype>5): return {'k':-1};

    fname,fname_base,fname_ext = check_if_filename_exists(sdir,snum,\
        snapshot_name=snapshot_name,extension=extension,four_char=four_char)

    if(fname=='NULL'): return {'k':-1}
    if(loud==1): print('loading file : '+fname)

    ## open file and parse its header information
    nL = 0 # initial particle point to start at 
    if(fname_ext=='.hdf5'):
        file = h5py.File(fname,'r') # Open hdf5 snapshot file
        header_master = file["Header"] # Load header dictionary (to parse below)
        header_toparse = header_master.attrs
    else:
        file = open(fname) # Open binary snapshot file
        header_toparse = load_gadget_binary_header(file)

    npart = header_toparse["NumPart_ThisFile"]
    massarr = header_toparse["MassTable"]
    time_var = header_toparse["Time"]
    redshift = header_toparse["Redshift"]
    flag_sfr = header_toparse["Flag_Sfr"]
    flag_feedbacktp = header_toparse["Flag_Feedback"]
    npartTotal = header_toparse["NumPart_Total"]
    flag_cooling = header_toparse["Flag_Cooling"]
    numfiles = header_toparse["NumFilesPerSnapshot"]
    boxsize = header_toparse["BoxSize"]
    #omega_matter = header_toparse["Omega0"]
    #omega_lambda = header_toparse["OmegaLambda"]
    hubble = header_toparse["HubbleParam"]
    flag_stellarage = header_toparse["Flag_StellarAge"]
    flag_metals = header_toparse["Flag_Metals"]
    newheader = [npart, massarr, time_var, redshift, flag_sfr, flag_feedbacktp, npartTotal, flag_cooling, numfiles, boxsize, hubble, flag_stellarage, flag_metals] #omega_matter, omega_lambda
    print("npart_file: ",npart)
    print("npart_total:",npartTotal)

    hinv=1.
    if (h0==1):
        hinv=1./hubble
    ascale=1.
    if (cosmological==1):
        ascale=time_var
        hinv=1./hubble
    if (cosmological==0): 
        time_var*=hinv
    
    boxsize*=hinv*ascale
    if (npartTotal[ptype]<=0): file.close(); return {'k':-1};
    if (header_only==1): file.close(); return {'k':0,'time':time_var,
        'boxsize':boxsize,'hubble':hubble,'npart':npart,'npartTotal':npartTotal};

    # initialize variables to be read

    pos=np.zeros([npartTotal[ptype],3],dtype=float)
    if (ptype==0):
        rho=np.zeros([npartTotal[ptype]],dtype=float)

    # loop over the snapshot parts to get the different data pieces
    for i_file in range(numfiles):
        
        if (numfiles>1):
            file.close()
            fname = fname_base+'.'+str(i_file)+fname_ext
            if(fname_ext=='.hdf5'):
                file = h5py.File(fname,'r') # Open hdf5 snapshot file
            else:
                file = open(fname) # Open binary snapshot file
                header_toparse = load_gadget_binary_header(file)
                
        if (fname_ext=='.hdf5'):
            input_struct = file
            npart = file["Header"].attrs["NumPart_ThisFile"]
            bname = "PartType"+str(ptype)+"/"
        else:
            npart = header_toparse['NumPart_ThisFile']
            input_struct = load_gadget_binary_particledat(file, header_toparse, ptype, skip_bh=skip_bh)
            bname = ''
            
        
        # now do the actual reading
        if(npart[ptype]>0):
            nR=nL + npart[ptype]
            print("Reading Positions",i_file+1,"/",numfiles)
            pos[nL:nR,:]=input_struct[bname+"Coordinates"]
            if (ptype==0):
                print("Reading Densities",i_file+1,"/",numfiles)
                rho[nL:nR]=input_struct[bname+"Density"]
            nL = nR # sets it for the next iteration	


    # do the cosmological conversions on final vectors as needed
    pos *= hinv*ascale # snapshot units are comoving
    if (ptype==0):
        rho *= (hinv/((ascale*hinv)**3))


    file.close();
    if (ptype==0):
        return {'k':1,'p':pos,'rho':rho, 'header':newheader}; #Return Density for gas to help with centering

    return {'k':1,'p':pos, 'header':newheader}


def readsnap_trunc(sdir,snum,ptype,truncMask,
    snapshot_name='snapshot',
    extension='.hdf5',
    h0=0,cosmological=0,skip_bh=0,four_char=0,
    header_only=0,loud=0):
    
    if (ptype<0): return {'k':-1};
    if (ptype>5): return {'k':-1};

    fname,fname_base,fname_ext = check_if_filename_exists(sdir,snum,\
        snapshot_name=snapshot_name,extension=extension,four_char=four_char)

    if(fname=='NULL'): return {'k':-1}
    if(loud==1): print('loading file : '+fname)

    ## open file and parse its header information
    nL = 0 # initial particle point to start at
    nL0 = 0
    if(fname_ext=='.hdf5'):
        file = h5py.File(fname,'r') # Open hdf5 snapshot file
        header_master = file["Header"] # Load header dictionary (to parse below)
        header_toparse = header_master.attrs
    else:
        file = open(fname) # Open binary snapshot file
        header_toparse = load_gadget_binary_header(file)

    npart = header_toparse["NumPart_ThisFile"]
    massarr = header_toparse["MassTable"]
    time_var = header_toparse["Time"]
    redshift = header_toparse["Redshift"]
    flag_sfr = header_toparse["Flag_Sfr"]
    flag_feedbacktp = header_toparse["Flag_Feedback"]
    npartTotal = header_toparse["NumPart_Total"]
    flag_cooling = header_toparse["Flag_Cooling"]
    numfiles = header_toparse["NumFilesPerSnapshot"]
    boxsize = header_toparse["BoxSize"]
    #omega_matter = header_toparse["Omega0"]
    #omega_lambda = header_toparse["OmegaLambda"]
    hubble = header_toparse["HubbleParam"]
    flag_stellarage = header_toparse["Flag_StellarAge"]
    flag_metals = header_toparse["Flag_Metals"]
    newheader = [npart, massarr, time_var, redshift, flag_sfr, flag_feedbacktp, npartTotal, flag_cooling, numfiles, boxsize,  hubble, flag_stellarage, flag_metals]#omega_matter, omega_lambda,
    print("npart_file: ",npart)
    print("npart_total:",npartTotal)

    hinv=1.
    if (h0==1):
        hinv=1./hubble
    ascale=1.
    if (cosmological==1):
        ascale=time_var
        hinv=1./hubble
    if (cosmological==0): 
        time_var*=hinv
    
    boxsize*=hinv*ascale
    if (npartTotal[ptype]<=0): file.close(); return {'k':-1};
    if (header_only==1): file.close(); return {'k':0,'time':time_var,
        'boxsize':boxsize,'hubble':hubble,'npart':npart,'npartTotal':npartTotal};

    # initialize variables to be read
    Ntrunc =  np.size(np.where(truncMask))
    mass=np.zeros(Ntrunc,dtype=float)
    #if (ptype==0) or (ptype==4):
    if True:
        vel=np.zeros([Ntrunc,3],dtype=float)

    if (ptype==0):
        ugas=np.copy(mass)
        hsml=np.copy(mass) 
        fH2 = np.copy(mass)
        if (flag_cooling>0): 
            nume=np.copy(mass)
            numh=np.copy(mass)
    if (ptype==0) and (flag_metals > 0):
        metal=np.zeros([Ntrunc,flag_metals],dtype=float)
    if (ptype==4) and (flag_sfr>0) and (flag_stellarage>0):
        stellage=np.copy(mass)

    # loop over the snapshot parts to get the different data pieces
    for i_file in range(numfiles):
        
        if (numfiles>1):
            file.close()
            fname = fname_base+'.'+str(i_file)+fname_ext
            if(fname_ext=='.hdf5'):
                file = h5py.File(fname,'r') # Open hdf5 snapshot file
            else:
                file = open(fname) # Open binary snapshot file
                header_toparse = load_gadget_binary_header(file)
                
        if (fname_ext=='.hdf5'):
            input_struct = file
            npart = file["Header"].attrs["NumPart_ThisFile"]
            bname = "PartType"+str(ptype)+"/"
        else:
            npart = header_toparse['NumPart_ThisFile']
            input_struct = load_gadget_binary_particledat(file, header_toparse, ptype, skip_bh=skip_bh)
            bname = ''
            
        
        # now do the actual reading
        
        if(npart[ptype]>0):
            nR0=nL0 + npart[ptype]
            nR=nL + np.size(np.where(truncMask[nL0:nR0]))
            if nR>nL:
             print("Reading Mass",i_file+1,"/",numfiles)
             mass[nL:nR]=(massarr[ptype])
             if (massarr[ptype] <= 0.):
                #mass[nL0:nR0] = input_struct[bname+"Masses"]
                tmp = np.zeros((nR0-nL0))
                tmp[:]=(input_struct[bname+"Masses"])
                mass[nL:nR]=tmp[truncMask[nL0:nR0]]
                del tmp
             #if (ptype==0) or (ptype==4):
             if True:
                print("Reading Velocities",i_file+1,"/",numfiles)
                tmp = np.zeros((nR0-nL0,3))
                tmp[:,:] = input_struct[bname+"Velocities"]
                vel[nL:nR,:] = tmp[truncMask[nL0:nR0]]
                del tmp

                #vel[nL:nR,:]=input_struct[bname+"Velocities"][truncMask[nL0:nR0],:] #Try splitting the dimensions up??
             if (ptype==0):
                print("Reading Internal Energy",i_file+1,"/",numfiles)
                tmp = np.zeros((nR0-nL0))
                tmp[:]=(input_struct[bname+"InternalEnergy"])
                ugas[nL:nR]=tmp[truncMask[nL0:nR0]]
                del tmp
                try:
                    fH2[nL:nR] = (input_struct[bname+"MolecularMassFraction"])[truncMask[nL0:nR0]]
                except:
                    fH2[nL:nR]-=1
                print("Reading SmoothingLength",i_file+1,"/",numfiles)
                tmp = np.zeros((nR0-nL0))
                tmp[:]=(input_struct[bname+"SmoothingLength"])
                hsml[nL:nR]=tmp[truncMask[nL0:nR0]]
                del tmp
                if (flag_cooling > 0): 
                    print("Reading Electron Abundance",i_file+1,"/",numfiles)
                    tmp = np.zeros((nR0-nL0))
                    tmp[:]=(input_struct[bname+"ElectronAbundance"])
                    nume[nL:nR]=tmp[truncMask[nL0:nR0]]
                    del tmp
                    print("Reading Neutral Hydrogen Abundance",i_file+1,"/",numfiles)
                    tmp = np.zeros((nR0-nL0))
                    tmp[:]=(input_struct[bname+"NeutralHydrogenAbundance"])
                    numh[nL:nR]=tmp[truncMask[nL0:nR0]]
                    del tmp
             if (ptype==0) and (flag_metals > 0):
                print("Reading Metallicity",i_file+1,"/",numfiles)
                #tmp = np.zeros((nR0-nL0,Nmetals))
                tmp=np.array(input_struct[bname+"Metallicity"])
                print("Shape of Metallicity File is:",np.shape(tmp))
                #if (flag_metals > 1):
                #    if (metal_t.shape[0] != npart[ptype]): 
                #        metal_t=np.transpose(metal_t)
                #else:
                #    metal_t=np.reshape(np.array(metal_t),(np.array(metal_t).size,1))
                #tmp = np.zeros(np.shape(metal_t))
                #if np.size(np.shape(metal_t))>1:
                #    tmp[:,:] = metal_t
                #else:
                #    tmp[:] = metal_t
                metal[nL:nR,:]=tmp[truncMask[nL0:nR0]]
                del tmp
             if (ptype==4) and (flag_sfr>0) and (flag_stellarage>0):
                print("Reading Stellar Age",i_file+1,"/",numfiles)
                tmp = np.zeros((nR0-nL0))
                tmp[:]=(input_struct[bname+"StellarFormationTime"])
                stellage[nL:nR]=tmp[truncMask[nL0:nR0]]
                del tmp
            nL = nR # sets it for the next iteration
            nL0 = nR0	


    # do the cosmological conversions on final vectors as needed
    mass *= hinv
    #if (ptype==0) or (ptype==4):
    if True:
        vel *= np.sqrt(ascale) # remember gadget's weird velocity units!
    if (ptype==0):
        hsml *= hinv*ascale
    if (ptype==4) and (flag_sfr>0) and (flag_stellarage>0) and (cosmological==0):
        stellage *= hinv

    file.close();
    if (ptype==0):
        return {'k':1,'v':vel,'m':mass,'u':ugas,'h':hsml,'ne':nume,'nh':numh,'z':metal, 'fH2':fH2, 'header':newheader};
    if (ptype==4):
        return {'k':1,'v':vel,'m':mass,'age':stellage, 'header':newheader}
    return {'k':1,'m':mass, 'v':vel, 'header':newheader}




def check_if_filename_exists(sdir,snum,snapshot_name='snapshot',extension='.hdf5',four_char=0):
    for extension_touse in [extension,'.bin','']:
        fname=sdir+'/'+snapshot_name+'_'
        ext='00'+str(snum);
        if (int(snum)>=10): ext='0'+str(snum)
        if (int(snum)>=100): ext=str(snum)
        if (four_char==1): ext='0'+ext
        if (int(snum)>=1000): ext=str(snum)
        fname+=ext
        fname_base=fname
        s0=sdir.split("/"); snapdir_specific=s0[len(s0)-1];
        if(len(snapdir_specific)<=1): snapdir_specific=s0[len(s0)-2];

        ## try several common notations for the directory/filename structure
        fname=fname_base+extension_touse;
        if not os.path.exists(fname): 
            ## is it a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap' instead of 'snapshot'?
            fname_base=sdir+'/snap_'+ext; 
            fname=fname_base+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap' instead of 'snapshot', AND its a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap(snapdir)' instead of 'snapshot'?
            fname_base=sdir+'/snap_'+snapdir_specific+'_'+ext; 
            fname=fname_base+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap' instead of 'snapshot', AND its a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is it in a snapshot sub-directory? (we assume this means multi-part files)
            fname_base=sdir+'/snapdir_'+ext+'/'+snapshot_name+'_'+ext; 
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is it in a snapshot sub-directory AND named 'snap' instead of 'snapshot'?
            fname_base=sdir+'/snapdir_'+ext+'/'+'snap_'+ext; 
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## wow, still couldn't find it... ok, i'm going to give up!
            fname_found = 'NULL'
            fname_base_found = 'NULL'
            fname_ext = 'NULL'
            continue;
        fname_found = fname;
        fname_base_found = fname_base;
        fname_ext = extension_touse
        break; # filename does exist! 
    return fname_found, fname_base_found, fname_ext;



def load_gadget_binary_header(f):
    ### Read header.
    import array
    # Skip 4-byte integer at beginning of header block.
    f.read(4)
    # Number of particles of each type. 6*unsigned integer.
    Npart = array.array('I')
    Npart.fromfile(f, 6)
    # Mass of each particle type. If set to 0 for a type which is present, 
    # individual particle masses from the 'mass' block are used instead.
    # 6*double.
    Massarr = array.array('d')
    Massarr.fromfile(f, 6)
    # Expansion factor (or time, if non-cosmological sims) of output. 1*double. 
    a = array.array('d')
    a.fromfile(f, 1)
    a = a[0]
    # Redshift of output. Should satisfy z=1/a-1. 1*double.
    z = array.array('d')
    z.fromfile(f, 1)
    z = float(z[0])
    # Flag for star formation. 1*int.
    FlagSfr = array.array('i')
    FlagSfr.fromfile(f, 1)
    # Flag for feedback. 1*int.
    FlagFeedback = array.array('i')
    FlagFeedback.fromfile(f, 1)
    # Total number of particles of each type in the simulation. 6*int.
    Nall = array.array('i')
    Nall.fromfile(f, 6)
    # Flag for cooling. 1*int.
    FlagCooling = array.array('i')
    FlagCooling.fromfile(f, 1)
    # Number of files in each snapshot. 1*int.
    NumFiles = array.array('i')
    NumFiles.fromfile(f, 1)
    # Box size (comoving kpc/h). 1*double.
    BoxSize = array.array('d')
    BoxSize.fromfile(f, 1)
    # Matter density at z=0 in units of the critical density. 1*double.
    Omega0 = array.array('d')
    Omega0.fromfile(f, 1)
    # Vacuum energy density at z=0 in units of the critical density. 1*double.
    OmegaLambda = array.array('d')
    OmegaLambda.fromfile(f, 1)
    # Hubble parameter h in units of 100 km s^-1 Mpc^-1. 1*double.
    h = array.array('d')
    h.fromfile(f, 1)
    h = float(h[0])
    # Creation times of stars. 1*int.
    FlagAge = array.array('i')
    FlagAge.fromfile(f, 1)
    # Flag for metallicity values. 1*int.
    FlagMetals = array.array('i')
    FlagMetals.fromfile(f, 1)

    # For simulations that use more than 2^32 particles, most significant word 
    # of 64-bit total particle numbers. Otherwise 0. 6*int.
    NallHW = array.array('i')
    NallHW.fromfile(f, 6)

    # Flag that initial conditions contain entropy instead of thermal energy
    # in the u block. 1*int.
    flag_entr_ics = array.array('i')
    flag_entr_ics.fromfile(f, 1)

    # Unused header space. Skip to particle positions.
    f.seek(4+256+4+4)

    return {'NumPart_ThisFile':Npart, 'MassTable':Massarr, 'Time':a, 'Redshift':z, \
    'Flag_Sfr':FlagSfr[0], 'Flag_Feedback':FlagFeedback[0], 'NumPart_Total':Nall, \
    'Flag_Cooling':FlagCooling[0], 'NumFilesPerSnapshot':NumFiles[0], 'BoxSize':BoxSize[0], \
    'Omega0':Omega0[0], 'OmegaLambda':OmegaLambda[0], 'HubbleParam':h, \
    'Flag_StellarAge':FlagAge[0], 'Flag_Metals':FlagMetals[0], 'Nall_HW':NallHW, \
    'Flag_EntrICs':flag_entr_ics[0]}


def load_gadget_binary_particledat(f, header, ptype, skip_bh=0):
    ## load old format=1 style gadget binary snapshot files (unformatted fortran binary)
    import array
    gas_u=0.; gas_rho=0.; gas_ne=0.; gas_nhi=0.; gas_hsml=0.; gas_SFR=0.; star_age=0.; 
    zmet=0.; bh_mass=0.; bh_mdot=0.; mm=0.;
    Npart = header['NumPart_ThisFile']
    Massarr = header['MassTable']
    NpartTot = np.sum(Npart)
    NpartCum = np.cumsum(Npart)
    n0 = NpartCum[ptype] - Npart[ptype]
    n1 = NpartCum[ptype]
    
    ### particles positions. 3*Npart*float.
    pos = array.array('f')
    pos.fromfile(f, 3*NpartTot)
    pos = np.reshape(pos, (NpartTot,3))
    f.read(4+4) # Read block size fields.

    ### particles velocities. 3*Npart*float.
    vel = array.array('f')
    vel.fromfile(f, 3*NpartTot)
    vel = np.reshape(vel, (NpartTot,3))
    f.read(4+4) # Read block size fields.

    ### Particle IDs. # (Npart[0]+...+Npart[5])*int
    id = array.array('i')
    id.fromfile(f, NpartTot)
    id = np.array(id)
    f.read(4+4) # Read block size fields.
        
    ### Variable particle masses. 
    Npart_MassCode = np.copy(np.array(Npart))
    Npart=np.array(Npart)
    Npart_MassCode[(Npart <= 0) | (np.array(Massarr,dtype='d') > 0.0)] = 0
    NwithMass = np.sum(Npart_MassCode)
    mass = array.array('f')
    mass.fromfile(f, NwithMass)
    f.read(4+4) # Read block size fields.
    if (Massarr[ptype]==0.0):
        Npart_MassCode_Tot = np.cumsum(Npart_MassCode)
        mm = mass[Npart_MassCode_Tot[ptype]-Npart_MassCode[ptype]:Npart_MassCode_Tot[ptype]]

    if ((ptype==0) | (ptype==4) | (ptype==5)):
        if (Npart[0]>0):
            ### Internal energy of gas particles ((km/s)^2).
            gas_u = array.array('f')
            gas_u.fromfile(f, Npart[0])
            f.read(4+4) # Read block size fields.
            ### Density for the gas paraticles (units?).
            gas_rho = array.array('f')
            gas_rho.fromfile(f, Npart[0])
            f.read(4+4) # Read block size fields.

            if (header['Flag_Cooling'] > 0):
                ### Electron number density for gas particles (fraction of n_H; can be >1).
                gas_ne = array.array('f')
                gas_ne.fromfile(f, Npart[0])
                f.read(4+4) # Read block size fields.
                ### Neutral hydrogen number density for gas particles (fraction of n_H).
                gas_nhi = array.array('f')
                gas_nhi.fromfile(f, Npart[0])
                f.read(4+4) # Read block size fields.

            ### Smoothing length (kpc/h). ###
            gas_hsml = array.array('f')
            gas_hsml.fromfile(f, Npart[0])
            f.read(4+4) # Read block size fields.

            if (header['Flag_Sfr'] > 0):
                ### Star formation rate (Msun/yr). ###
                gas_SFR = array.array('f')
                gas_SFR.fromfile(f, Npart[0])
                f.read(4+4) # Read block size fields.

        if (Npart[4]>0):
            if (header['Flag_Sfr'] > 0):
                if (header['Flag_StellarAge'] > 0):
                    ### Star formation time (in code units) or scale factor ###
                    star_age = array.array('f')
                    star_age.fromfile(f, Npart[4])
                    f.read(4+4) # Read block size fields.
        
        if (Npart[0]+Npart[4]>0):
            if (header['Flag_Metals'] > 0):
                ## Metallicity block (species tracked = Flag_Metals)
                if (Npart[0]>0):
                    gas_z = array.array('f')
                    gas_z.fromfile(f, header['Flag_Metals']*Npart[0])
                if (Npart[4]>0):
                    star_z = array.array('f')
                    star_z.fromfile(f, header['Flag_Metals']*Npart[4])
                f.read(4+4) # Read block size fields.
                if (ptype==0): zmet=np.reshape(gas_z,(-1,header['Flag_Metals']))
                if (ptype==4): zmet=np.reshape(star_z,(-1,header['Flag_Metals']))
        
        if (Npart[5]>0):
            if (skip_bh > 0):
                ## BH mass (same as code units, but this is the separately-tracked BH mass from particle mass)
                bh_mass = array.array('f')
                bh_mass.fromfile(f, Npart[5])
                f.read(4+4) # Read block size fields.
                ## BH accretion rate in snapshot
                bh_mdot = array.array('f')
                bh_mdot.fromfile(f, Npart[5])
                f.read(4+4) # Read block size fields.
    
    return {'Coordinates':pos[n0:n1,:], 'Velocities':vel[n0:n1,:], 'ParticleIDs':id[n0:n1], \
        'Masses':mm, 'Metallicity':zmet, 'StellarFormationTime':star_age, 'BH_Mass':bh_mass, \
        'BH_Mdot':bh_mdot, 'InternalEnergy':gas_u, 'Density':gas_rho, 'SmoothingLength':gas_hsml, \
        'ElectronAbundance':gas_ne, 'NeutralHydrogenAbundance':gas_nhi, 'StarFormationRate':gas_SFR}
        
        
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

    hf.close()
    return r0,posCenter,Lhat,velCenter