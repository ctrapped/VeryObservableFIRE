import numpy as np
import time
import argparse

from MakeDataset import MakeDataset
from EmissionSpecies import GetEmissionSpeciesParameters
from LoadParamFile import LoadParams

####Runs VeryObservableFIRE to create synthetic images from given observational parameters.
####Also creates corresponding projected and deprojected maps of radial mass flux, rotational velocity, and mass for the purposes of neural network training
####To use, modify param_template.param and pass the path to it as the first argument.
####    e.g. python VeryObservableFIRE.py param_template.param
####Optional overrides (any order): --galName, --minSnap, --maxSnap, --inclination (see --help)
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/08/2023



pi = np.pi
arcsec = (1. /60. / 60.) * pi/180.
c_km_s = 3*10**5 #speed of light in km/s
h = 4.135667696*np.power(10.,-15.) #eV * s
startTime=time.time()

def parse_args():
    parser = argparse.ArgumentParser(description="Run VeryObservableFIRE to generate synthetic images/annotations from a parameter file.")
    parser.add_argument("paramFile", help="Path to the .param file (e.g. param_template.param)")
    parser.add_argument("--galName", default=None, help="Override galName from the parameter file")
    parser.add_argument("--minSnap", type=int, default=None, help="Override the starting snapshot number")
    parser.add_argument("--maxSnap", type=int, default=None, help="Override the ending snapshot number (defaults to minSnap if omitted)")
    parser.add_argument("--inclination", type=int, default=None, help="Restrict to a single inclination, overriding the inclinations list in the parameter file")
    return parser.parse_args()

args = parse_args()
if args.maxSnap is None:
    args.maxSnap = args.minSnap

#`config` is passed down through MakeDataset/GenerateSyntheticImage/ProjectImage as a single dict;
#each of those functions only takes the (snapshot, inclination, position_angle, ...) identity/loop
#arguments explicitly and reads everything else out of config.
config = LoadParams(args.paramFile, galName=args.galName, minSnap=args.minSnap, maxSnap=args.maxSnap, inclination=args.inclination)

galName = config['galName']
minSnap = config['minSnap']
maxSnap = config['maxSnap']
snapStep = config['snapStep']
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

for Nsnap in range(minSnap,maxSnap+1,snapStep):
    print(Nsnap)
    MakeDataset(config, fileDir, statsDir, Nsnap, output, galName)

print("Time to finish: ",time.time()-startTime)
