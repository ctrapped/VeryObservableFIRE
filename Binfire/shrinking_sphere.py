import numpy as np
import h5py as h5py
import os.path
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import math

####Functions to perform the shrinking sphere fit
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/14/2023

def center_of_mass(Smas,r_s):
   #Simple center of mass calculation with output
   pos_com=np.zeros((3))
   pos_com[:] = np.divide(  np.dot(np.transpose(r_s),Smas) , np.sum(Smas)  )
   print("New center of mass:",pos_com)
   return pos_com
  

def shrink_sphere(Smas,Spos,wiggle_count,r=None,rmin=10,shrink_factor=0.7,old_com = None):
    #Shrinking Sphere algorithm to find center of mass. Calculates the center of mass within a sphere of radius r,
    #then shrinks the sphere by a factor of shrink_factor. Continues until r<=rmin and the center of mass does not change
    #by more than delta_cutoff between iterations. Recommended to use stellar mass, at least for m12'scipy
    
    delta_cutoff = 0.000001
    NS = np.size(Smas)
    print(NS," Stars in sphere.")
   
    print("Calculating new center of mass...")
    pos_com = center_of_mass(Smas,Spos)


    r2_S = np.add(np.power(Spos[:,0]-pos_com[0],2),np.add(np.power(Spos[:,1]-pos_com[1],2),np.power(Spos[:,2]-pos_com[2],2)))

    r_new = r*shrink_factor 
    print("Rnew is",r_new  )
   
    if r_new > rmin: #iterate until few enough particles
#Set a min radius, #Iterate at the minimum radius until the com is stable
        print("Iterating again...")
        Sidx = np.where(r2_S<np.power(r_new,2))
        print(Sidx[0])
        del r2_S

        final_com = shrink_sphere(Smas[Sidx],Spos[Sidx],wiggle_count,r_new,rmin,shrink_factor,pos_com)
    
    else:
        Sidx = np.where(r2_S<rmin**2)
        del r2_S

       
        final_com = center_of_mass(Smas[Sidx],Spos[Sidx])
        delta_com = np.subtract(final_com,old_com)
        delta = np.linalg.norm(delta_com)
        if ((delta > delta_cutoff) and (wiggle_count < 1000)) or ((wiggle_count < 10) and (delta != 0.0)):
            print("Wiggling the sphere around...")
            print("Delta C.O.M. =",delta)
            wiggle_count+=1
            final_com = shrink_sphere(Smas[Sidx],Spos[Sidx],wiggle_count,rmin,rmin,shrink_factor,final_com)
        else:
            print("Delta C.O.M. =",delta)
            print("Done with shrinking sphere...")
            print("Wiggled ",wiggle_count," times.")

    return final_com
        