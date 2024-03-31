import numpy as np
from VOF_EmissionSpecies import GetEmissionSpeciesParameters

####Functions to generate emission/absorption for particles along a sightline in order to construct the mock spectra
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023

#gamma_ul is the sum of all A_ul's and A_lu's for all possible emissions from u and all possible absorptions from l (pg 57 in Draine)

unit_M = 10**10 * 1.98855 *10**33 #10^10 solar masses / h !!in grams!! #h accounted for in readsnap
unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
unit_V = 1.0*10.0**5 # 1 km/s !!in cm/s!! Converted by factor of sqrt(a) in readsnap
unit_T = unit_L/unit_V
unit_rho = unit_M / unit_L**3

proton_mass = 1.6726219*10.**(-27.)*(1000.0/unit_M) ##appropriate units

h = 4.135667696*np.power(10.,-15.) #eV * s
m_e = 9.1094*np.power(10.,-28.) #grams
c = 3.*np.power(10.,10.) #cm/s
kb = 8.617333262145*np.power(10.,-5.) # eV * K-1
pi = np.pi
e = 4.8032*np.power(10.,-10.) #cm^(3/2) * g^(1/2) * s^(-1)
amu = 1.6735575*np.power(10.,-24) / unit_M #grams to unit mass

kpc2cm = 3.086*np.power(10.,21.)

pi = np.pi
eps = 1e-10
arcsec2rad = pi / (180.*3600.)
defaultRes = 1. * arcsec2rad #default beamsize of 1 arcsec


def GenerateSpectra(gMass,speciesMassFrac,dopplerVelocity,particleSize,temp,distance,impact,species,beamSize,Nspec,bandwidth,calcThermalLevels):

    mass_species,g_upper,g_lower,E_upper,E_lower,A_ul,gamma_ul,Glevels,Elevels,n_u_fraction,n_l_fraction = GetEmissionSpeciesParameters(species) #Load the emission/absorption parameters for this species
    N_molecules = np.multiply(gMass , speciesMassFrac) / mass_species
    if np.size(N_molecules)>0 and np.min(N_molecules)<0:
        print("Warning: N_molecules<0")
        print("    gMass=",np.min(gMass))
        print("    speciesMassFrac = ",np.min(speciesMassFrac))
        print("    mass_species = ",mass_species)

    beamRadiusPhysical = distance * np.sin(beamSize) / 2. #sim units (kpc)



    colDens,pathLength = GetColumnDensityAlongLOS(particleSize, impact,N_molecules,beamRadiusPhysical) #kpc^-2
    upperToLowerRate,attenuationCrossSection = GenEmissionAndAbsorptionRates(colDens,temp,species,beamRadiusPhysical,calcThermalLevels) #Units of Hz
    nu_ul = (E_upper-E_lower)/h #in Hz
    f_lu = g_upper / g_lower * A_ul * (m_e * c**3.)/(8.*pi**2.*e**2.*nu_ul**2.) #unitless

    nu_0 = nu_ul
    spectralRange = [nu_0-bandwidth/2. , nu_0+bandwidth/2.]
     
    #Actually generate the spectrum with the above parameters
    return MakeSpectrum(upperToLowerRate,attenuationCrossSection,distance,dopplerVelocity,temp,colDens,pathLength,nu_ul,f_lu,gamma_ul,spectralRange,mass_species,beamRadiusPhysical,Nspec,species)


    
def GetColumnDensityAlongLOS(r,impact,N_mol_in_particle,beamRadiusPhysical):
    #calculate an effect path length through the material (i.e. the particles are spheres and the beam is of finite size, so each part of the beam will not pass through the same length)
    #This can be improved upon, as it is an approximation of what the FIRE simulations actually represent
    vol = 4/3 * pi * np.power(r,3)
    chordLength = CalcEffectiveChordLength(r,impact,beamRadiusPhysical)
    colDensParticle = np.multiply(np.divide(N_mol_in_particle , vol) , chordLength)
    if np.size(colDensParticle)>0 and np.min(colDensParticle)<0:
        print("Warning: colDensParticle<0, min=",np.min(colDensParticle))
        print("    N_mol_in_particle = ",np.min(N_mol_in_particle))
        print("    vol = ",np.min(vol))
        print("    chordLength = ",np.min(chordLength))
    return colDensParticle,chordLength         

def CalcEffectiveChordLength(r,impact,beamRadiusPhysical):
    #Finds an average chord length through the sphere for a given beam width
    Nsample = 100

    impactMax = impact+beamRadiusPhysical
    impactMin = impact-beamRadiusPhysical

    impactMax[np.greater(impactMax,r)] = r[np.greater(impactMax,r)]
    impactMin[np.less(impactMin,-r)] = -r[np.less(impactMin,-r)]


    impactInc = (impactMax-impactMin)/Nsample
    chordLength = np.zeros((np.size(r)))
    for i in range(0,Nsample):
        impact = np.abs(impactMin+i*impactInc)
        mask = np.where((r-impact)>0)
        chordLength[mask] += 2*np.multiply( r[mask],np.sin(np.arccos( np.divide(impact[mask] , r[mask]) )) ) / Nsample #calculates an effective path length through the particle

    return chordLength

def MakeSpectrum(upperToLowerRate,attenuationCrossSection,distance,dopplerVelocity,temp,colDens,pathLength,nu_ul,f_lu,gamma_ul,spectralRange,mass_species,beamRadiusPhysical,Nspec,species):
    #Define min/max frequency of spectra somehow
    beamAreaPhysical = pi*np.power(beamRadiusPhysical,2)
    Nparticles = np.size(upperToLowerRate);
    spectra = np.zeros((Nspec))
    emission = np.copy(spectra)
    optical_depth = np.copy(emission)
    nu = np.linspace(spectralRange[0],spectralRange[1],Nspec) 

    for p in range(0,Nparticles): #This form must be executed sequentially, as absorption is built up. Particles are sorted based on distance along sightline
        sigma_nu = GenLineProfile(nu_ul, f_lu, gamma_ul, dopplerVelocity[p], spectralRange, temp[p], mass_species, Nspec)
        phi_nu = sigma_nu * m_e * c / pi / (e**2) / f_lu
        if distance[p]==0:
            distance[p]=eps;

        
        power_emission = (upperToLowerRate[p] * h) * np.multiply(nu,phi_nu)

        emission += power_emission/ distance[p]**2
        spectra = np.add(spectra , np.multiply(power_emission , np.exp(-optical_depth)) / distance[p]**2)
        
        optical_depth_particle = phi_nu*attenuationCrossSection[p]
        if np.min(optical_depth_particle)<0:
            print("WARNING, NEGATIVE OPTICAL DEPTH -> ",np.min(optical_depth_particle)," : ",np.max(optical_depth_particle))
            print ("min(phi_nu)=",np.min(phi_nu))
            print("attenuationCrossSection[p]=",attenuationCrossSection[p])
        
        optical_depth_particle[optical_depth_particle<0]=0
        optical_depth += optical_depth_particle #Build up optical depth as you go through particles in the sightline

    return spectra,emission,optical_depth,nu



def GenLineProfile(nu_ul,f_lu,gamma_ul,dopplerVelocity,spectralRange,temp,mass_species,Nspec):
    #Defines line profile with a gaussian core and damping wings. Following Draine "Physics of the Interstellar and Intergalactic Medium"
    nu = np.linspace(spectralRange[0],spectralRange[1],Nspec) 

    spectral_resolution = np.abs(spectralRange[1]-spectralRange[0])/Nspec
    dopplerBeta = dopplerVelocity / (3.0*np.power(10,5)) #both in km/s

    doppler_freq_shift = nu_ul * (np.sqrt( np.divide(-dopplerBeta+1 , dopplerBeta+1) ) - 1)
    doppler_index_shift = int(np.round( doppler_freq_shift / spectral_resolution ))


    v = np.divide( 1 - np.power(nu / (nu_ul+doppler_freq_shift) , 2) , 1 + np.power(nu / (nu_ul+doppler_freq_shift) , 2)) * c #in cm/s

    lamda_ul = c / nu_ul 

    b=12.90*np.sqrt(temp*np.power(10.,-4) / (mass_species/amu))*1000.*100. #cm/s
    b6 = b / (10.*1000.*100.) #b/10 km/s unitless
    
    prefactor = np.sqrt(pi)  * (e**2)/(m_e*c) * f_lu*lamda_ul / b 

    z = np.sqrt(10.31+np.log(7616 / (gamma_ul*lamda_ul)*b6)) 
    transition_v = np.abs(b*z)

    coreFreqs =  np.where(np.abs(v)<=transition_v)[0] 
    wingFreqs =  np.where(np.abs(v)> transition_v)[0]

    coreFreqs = coreFreqs[((coreFreqs<np.size(nu)) & (coreFreqs>=0))]

    wingFreqs = wingFreqs[((wingFreqs<np.size(nu)) & (wingFreqs>=0))]


    sigma = np.zeros(np.shape(nu))
    sigma[coreFreqs] = prefactor * np.exp(-np.power(v[coreFreqs],2) / b**2)
    sigma[wingFreqs] = np.divide(prefactor * (1/(4*np.power(pi,1.5)) * gamma_ul*lamda_ul * b) , np.power(v[wingFreqs],2)) #Was commented out? double check
    

    return sigma


    

def GenEmissionAndAbsorptionRates(colDens,T,species,beamRadiusPhysical,calcThermalLevels):
    #Find rates of emission and attenuation cross section. Following Draine "Physics of the Interstellar and Intergalactic Medium"
    mass_species,g_u,g_l,E_u,E_l,A_ul,gamma_ul,glevels,Elevels,n_u_fraction,n_l_fraction = GetEmissionSpeciesParameters(species)


    n_total = pi * np.multiply(colDens , np.power(beamRadiusPhysical,2)) #unitless
    
    #v=(E_u-E_l)/h #Hz
    
    Nlevels = np.size(Elevels)
    partitionFunction = np.zeros((np.shape(colDens)))
    for l in range(0,Nlevels):
        toAdd = glevels[l] * np.exp( np.divide(-(Elevels[l]-Elevels[0]) , kb*T) ) #Unitless
        partitionFunction += toAdd
        
    if n_u_fraction is None or calcThermalLevels:
        n_u = np.multiply(n_total  , np.divide( np.exp( np.divide(-(E_u-Elevels[0]) , kb*T)) * g_u , partitionFunction)) #unitless
    else:
        n_u = n_u_fraction * n_total #For species not in thermal equilibrium with entire particle

    if n_l_fraction is None or calcThermalLevels:
        n_l = np.multiply(n_total  , np.divide( np.exp( np.divide(-(E_l-Elevels[0]) , kb*T)) * g_l , partitionFunction)) #unitless
        colDensLower = np.multiply(colDens  , np.divide( np.exp( np.divide(-(E_l-Elevels[0]) , kb*T)) * g_l , partitionFunction)) / np.power(kpc2cm,2) #units cm^-2
    else:
        n_l = n_l_fraction * n_total
        colDensLower = n_l_fraction * colDens / np.power(kpc2cm,2) #units cm^-2

    nu_ul = (E_u-E_l)/h #in Hz
    B_ul = np.power(c,3) / (8*np.pi*h* np.power(nu_ul,3)) * A_ul
    B_lu = g_u / g_l * B_ul
    attenuationCrossSection = h * nu_ul / c * (colDensLower * B_lu)# - n_u * B_ul) Ignore stimulated emission for now
    if np.size(attenuationCrossSection)>0 and np.min(attenuationCrossSection)<0:
        print("Warning, attenuationCrossSection<0...")
        print("    colDensLower=",np.min(colDensLower))
        print("    B_lu=",B_lu)
        print("    colDens=",np.min(colDens))


    upperToLowerRate = n_u*A_ul# * (1+nbar_gamma) #Hz

    return upperToLowerRate , attenuationCrossSection
