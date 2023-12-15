import numpy as np

unit_M = 10**10 * 1.98855 *10**33 #10^10 solar masses / h !!in grams!! #h accounted for in readsnap
unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
unit_V = 1.0*10.0**5 # 1 km/s !!in cm/s!! Converted by factor of sqrt(a) in readsnap
unit_T = unit_L/unit_V
unit_rho = unit_M / unit_L**3

proton_mass = 1.6726219*10**(-27)*(1000.0/unit_M) ##appropriate units

pi = np.pi
eps = 0.00000000000000000000000000000000000000000000001
atomsPerMol = 6.02214*np.power(10.,23.) #Avagadro's Number

####Functions to read in emission species parameters for generating emission/absorption
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023


def GetEmissionSpeciesParameters(speciesName):
    n_u_fraction = None
    n_l_fraction = None
    if speciesName=="h_alpha":
    	#n = 3 ---> n = 2
        species_mass = 1.6735575*np.power(10.,-24) / unit_M #grams to unit mass
        g_upper = 9.
        g_lower = 4.
        E_upper = -1.5  #leave as eV for now
        E_lower = -3.4
        A_21 = 6.2649*np.power(10.,8) + 4.6986*np.power(10.,8) #Hz
        A_31 = (1.6725 + 0.55751 + 1.6725+6.2648) * np.power(10.,8) 
        A_32 = 4.410 * np.power(10,7) #Hz
        gamma_ul = A_32 + A_31 + A_21
        A_ul = A_32
        Nlevels = 100
        E0 = 13.6 #leave as eV for now
        Glevels = np.power( np.array(range(1,Nlevels)) , 2)
        Elevels = np.divide(-E0,Glevels)

    elif speciesName=='HI_21cm':
        #Hyperfine splitting
        species_mass = 1.6735575*np.power(10.,-24) / unit_M #grams to unit mass
        g_upper = 3
        g_lower = 1
        E_upper = -13.6 + 5.87433 * np.power(10.,-6.) #eV
        E_lower = -13.6 
        A_10 = 2.85 * np.power(10.,-15.) #Hz
        gamma_ul = A_10 #Need to consider other transitions??
        A_ul = A_10
        Elevels = np.array([E_lower , E_upper])
        Glevels = np.array([g_lower , g_upper]) #Safe to simplify as 2 state system? Gas emitting this should be cold anyway. May need to expand on this if hot gas emitting too much?
        n_u_fraction = 0.75
        n_l_fraction = 0.25


    elif speciesName=='NaI_D1':
        #2p^6 3p - > 2p^6 3s 5889.9
        species_mass = 22.98977 / atomsPerMol / unit_M #amu to g to unit mass
        g_upper = 3
        g_lower = 1
        E_upper = 2.102297159
        E_lower = 0
        A_ps = 6.16*np.power(10.,7)

        gamma_ul = A_ps # s is ground state, p is first excited state
        A_ul = A_ps
        Elevels,Glevels = ReadNistEnergyLevels("EnergyLevels/NIST_NaI_EnergyLevels.txt")
    elif speciesName=='NaI_D2':
        #2p^6 3p - > 2p^6 3s 5889.9
        species_mass = 22.98977 / atomsPerMol / unit_M #amu to g to unit mass
        g_upper = 3
        g_lower = 1
        E_upper = 2.104429184
        E_lower = 0
        A_ps = 6.14*np.power(10.,7)

        gamma_ul = A_ps # s is ground state, p is first excited state
        A_ul = A_ps
        Elevels,Glevels = ReadNistEnergyLevels("EnergyLevels/NIST_NaI_EnergyLevels.txt")
    else:
    	print("Error: Parameters for "+speciesName+" have not been defined.")
    	return 0,0,0,0,0,0,0,0,0,0,0

    return species_mass,g_upper,g_lower,E_upper,E_lower,A_ul,gamma_ul,Glevels,Elevels,n_u_fraction,n_l_fraction



def ReadNistEnergyLevels(filedir):
    fid = open(filedir,'r')
    fullFile= fid.read()
    lines = fullFile.split('\n')

    Elevels = np.zeros((np.size(lines)-1))
    Glevels = np.copy(Elevels)

    i=-1
    for line in lines:
        entries = line.split('\t')

        if (i>-1):
          if (np.size(entries)>3):

            energy=entries[3]
            energy = energy[1:len(energy)-1]

            if energy[0]=='[':
                energy = energy[1:len(energy)-1]
            i+=1
            Elevels[i] = float(energy)
            Glevels[i] = 1 #should just list degenerate energy levels multiple times
        else:
            i+=1


    Elevels = Elevels[Glevels>0]
    Glevels = Glevels[Glevels>0]

    return Elevels,Glevels
    

ReadNistEnergyLevels("EnergyLevels/NIST_NaI_EnergyLevels.txt")


       
        
