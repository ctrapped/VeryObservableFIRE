import numpy as np
import h5py as h5py
from Binfire import RunBinfire

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.colors import LogNorm



snapdir = '..\\..\\FIRE_Simulations\\'+galName+'_cr700_torqueRerun_01312023\\output\\snapdir_'
statsDir =  '..\\..\\FIRE_Simulations\\'+galName+'_cr700\\stats\\'+galName+'_cr700_stats_'
Nsnap = 591
output = None
maxima = [30,30,10]
res = [0.5,0.5,0.5]


binnedMass, binnedMassFlux = RunBinfire(snapdir,statsDir,Nsnap,output,maxima,res)

plt.imshow(np.sum(binnedMass,axis = 2),cmap='seismic',norm=LogNorm())
plt.show()

plt.close()
