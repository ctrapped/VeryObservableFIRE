from Binfire.shrinking_sphere import shrink_sphere
import numpy as np

unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
eps = 1e-10


####Various Centering functions used in Binfire
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 12/14/2023

def FindCenter(pos_center,Gdens,Gpos,Spos,Smas,rshrinksphere,rminsphere,shrinkfactor,shrinking_sphere_flag,old_a=None,v=None,ascale=None,mask_centers=None,mask_radius=None):
    #Function to determine the central position of a galaxy using a shrinking sphere algorithm. Also contains functionality to estimate where the center would be from a previous snapshot
    #
    #Inputs:
    #### pos_center : Initial guess of central position. Will default to the position of maximum density if left as None
    #### Gdens : Density of the gas particles
    #### Gpos : position of the gas particles
    #### Spos : position of the star particles
    #### Smas : mass of the start particles
    #### rshrinksphere : initial radius of the shrinking sphere in kpc
    #### shrinkfactor : how much the sphere is reduced each iteration
    #### shrinking_sphere_flag : set to 0 to skip shrinking sphere and just go with density estimate
    #### old_a : scale factor of next snapshot for use in estimating center at this snapshot
    #### v : central velocity of next snapshot for use in estimating center at this snapshot
    #### ascale : scale factor of this snapshot
    #### mask_centers : list of positions to mask. Use to ignore halos
    #### mask_radius : list of mask radii. Use to ignore halos
    N = np.size(Gdens)
    NS = np.size(Smas)
    if ((mask_centers is not None) and (mask_radius is not None)): #Mask data to ignore a certain halo/galaxy
        Nmask = np.size(mask_centers)/3 #Number of mask centers
        print("Nmask is",Nmask)
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
 
        
        


    if ((old_a is not None) and (v is not None) and (ascale is not None)):
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
    
    
    

def FindAngularMomentum(nh,Gpos,Gmom,Gtemp,r2):
    #Function to find orientation of the galaxy based on the average angular momentum vector of the cold dense gas
    #Inputs
    #### nh : Hydrogen number density
    #### Gpos : Position of gas particles
    #### Gmom : Momentum vector of gas particles
    #### Gtemp : Temperature of gas particles
    #### r2 : radial distance of each gas particle from the galactic center squared
    
    N = np.size(Gtemp)
    L = np.zeros((N,3)) #Initialize Angular Momentum Vector

    print("Finding orientation of disk...")
    L_total = np.zeros((3)) #sum of momentums in center
    L_avg = np.zeros((3)) #average momentum of central disk

    T_cutoff_L = 8000#Kelvin
    dens_cutoff_L = 1*unit_L**3#cm^-3
    r_cutoff_L = 10.0 #kpc for finding the average velocity of the galaxy (averaged over a sphere of radius r_cutoff)
    L = np.cross(Gpos[(Gtemp < T_cutoff_L) & (nh > dens_cutoff_L) & (r2 < r_cutoff_L**2)],Gmom[(Gtemp < T_cutoff_L) & (nh > dens_cutoff_L) & (r2 < r_cutoff_L**2)])

    L_avg[0] = np.mean(L[:,0])
    L_avg[1] = np.mean(L[:,1])
    L_avg[2] = np.mean(L[:,2])

    L_avg_mag = np.linalg.norm(L_avg)
    Lhat = np.zeros((3))
    Lhat[:] = np.divide(L_avg,L_avg_mag)
    return Lhat
    

def center_of_mass_velocity(mas,vel):
    mtot = np.sum(mas)
    v_com = np.zeros((3))
    v_com[0] = np.dot(mas,vel[:,0])/mtot
    v_com[1] = np.dot(mas,vel[:,1])/mtot
    v_com[2] = np.dot(mas,vel[:,2])/mtot
    return v_com



