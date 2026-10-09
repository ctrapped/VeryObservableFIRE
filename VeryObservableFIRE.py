import numpy as np
import time
import sys

from MakeDataset import MakeDataset
from EmissionSpecies import GetEmissionSpeciesParameters
from LoadParamFile import LoadParams

####Runs VeryObservableFIRE to create synthetic images from given observational parameters.
####Also creates corresponding projected and deprojected maps of radial mass flux, rotational velocity, and mass for the purposes of neural network training
####To use, modify param_template.param and pass the path to it as the first argument.
####    e.g. python VeryObservableFIRE.py param_template.param
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023



pi = np.pi
arcsec = (1. /60. / 60.) * pi/180.
c_km_s = 3*10**5 #speed of light in km/s
h = 4.135667696*np.power(10.,-15.) #eV * s
startTime=time.time()

paramFile = sys.argv[1]

try:
    galName = sys.argv[2]
except:
    galName=None

try:
    minSnap = int(sys.argv[3])
except:
    minSnap=None

try:
    maxSnap = int(sys.argv[4])
except:
    maxSnap=minSnap

try:
    inclination = int(sys.argv[5])
except:
    inclination=None

#`config` is passed down through MakeDataset/GenerateSyntheticImage/ProjectImage as a single dict;
#each of those functions only takes the (snapshot, inclination, position_angle, ...) identity/loop
#arguments explicitly and reads everything else out of config.
config = LoadParams(paramFile, galName=galName, minSnap=minSnap, maxSnap=maxSnap, inclination=inclination)

galName = config['galName']
minSnap = config['minSnap']
maxSnap = config['maxSnap']
fileDir = config['fileDir']
statsDir = config['statsDir']
output = config['output']

print("Looking at:"+fileDir)

config['targetBeamSize'] = config['targetBeamSize'] * arcsec #Convert from arcseconds (as given in the param file) to radians
config['bandwidth_km_s'] = config['Nchannels'] * config['res_km_s']
config['beamSize'] = 2*config['maxRadius']/config['Nsightlines1d'] / config['observerDistance']

mass_species,g_upper,g_lower,E_upper,E_lower,A_ul,gamma_ul,Glevels,Elevels,n_u_fraction,n_l_fraction = GetEmissionSpeciesParameters(config['speciesToRun'])
f0 = (E_upper-E_lower)/h #in Hz
config['Nspec'] = int(np.ceil(config['bandwidth_km_s'] / config['res_km_s']))
config['bandwidth'] = f0*c_km_s * (1 / (c_km_s-config['bandwidth_km_s']/2) - 1 / (c_km_s+config['bandwidth_km_s']/2))



print("##################    Calculated Observation Parameters    ###############")
print("Beamsize = ",config['beamSize'])
print("f0 = ",f0)
print("Bandwidth = ",config['bandwidth'])
print("###########################################################################")

for Nsnap in range(minSnap,maxSnap+1):
    print(Nsnap)
    MakeDataset(config, fileDir, statsDir, Nsnap, output, galName)

print("Time to finish: ",time.time()-startTime)
