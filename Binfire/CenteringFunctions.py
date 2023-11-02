from shrinking_sphere import shrink_sphere
from numpy import np

unit_L = 3.086*10**21 #1 kpc / h !!in cm!!
eps = 1e-10

def FindCenter(pos_center,Gdens,Gpos,Spos,Smas,rshrinksphere=5000,rminsphere=10,shrinkfactor=0.7,shrinking_sphere_flag=1):
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

    N = np.size(Gdens)
    NS = np.size(Smas)


        
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
    
    
    
def OrientGalaxy(pos,vel,Lhat,r0,returnPhi=False):
    #Function to transform positions and velocity vectors to a coordinate system with the z direction defined by the angular momentum vector Lhat and the x direction defined by arbitrary direction r0. Probably simpler and more elegant ways to rotate these matrices...
    #Inputs:
    #### pos: position of particles to orient (Should be centered already)
    #### vel: velocity of particles to orient
    #### Lhat: Vector to define the zhat direction in the new coordinate system 
    #### r0: Vector to define the xhat direction in the new coordinate system. Should be perpindicular to zhat
    #### returnPhi: Return phi coordinate
    
    N,dim = np.shape(pos)
    r_z = np.zeros((N,3))
    r_s = np.zeros((N,3))

    smag = np.zeros((N))
    zmag = np.zeros((N))

    zmag = np.dot(pos,Lhat)
    r_z[:,0] = zmag*Lhat[0]
    r_z[:,1] = zmag*Lhat[1]
    r_z[:,2] = zmag*Lhat[2]

    r_s = np.subtract(pos,r_z)
    smag = VectorArrayMag(r_s)
    smag[smag==0] = eps #make zero entries epsilon for division purposes

    s_hat = np.zeros((N,3))
    t_hat = np.zeros((N,3))

    s_hat[:,0] = np.divide(r_s[:,0],smag)
    s_hat[:,1] = np.divide(r_s[:,1],smag)
    s_hat[:,2] = np.divide(r_s[:,2],smag)
    t_hat = np.cross(Lhat,s_hat)

    phi = np.zeros((N))

    acos_term = np.divide(np.dot(r_s,r0),(np.linalg.norm(r0)*smag))
    acos_term[acos_term>1] = 1 #make sure the term isn't above magnitude 1 by a rounding error
    acos_term[acos_term<-1] = -1
    phi = np.multiply( np.arccos(acos_term) , np.sign(np.dot(np.cross(r0,r_s),Lhat))) #first term gets us |phi| from 0 to pi, second term gives us the sign
    del acos_term

    phi[phi<0] = phi[phi<0] + 2*pi #make all values range from 0 to pi

    xmag = np.multiply(smag,np.cos(phi))
    ymag = np.multiply(smag,np.sin(phi))

    xhat = r0
    zhat = Lhat
    yhat = np.cross(zhat,xhat)

    pos[:,0] = xmag
    pos[:,1] = ymag
    pos[:,2] = zmag

    if vel is not None:
        vel_tmp = np.copy(vel)
        vel[:,0] = np.multiply(vel_tmp[:,0],xhat[0]) + np.multiply(vel_tmp[:,1],xhat[1]) + np.multiply(vel_tmp[:,2],xhat[2]);
        vel[:,1] = np.multiply(vel_tmp[:,0],yhat[0]) + np.multiply(vel_tmp[:,1],yhat[1]) + np.multiply(vel_tmp[:,2],yhat[2]);
        vel[:,2] = np.multiply(vel_tmp[:,0],zhat[0]) + np.multiply(vel_tmp[:,1],zhat[1]) + np.multiply(vel_tmp[:,2],zhat[2]);

        if returnPhi:
            return pos,vel,phi
        else:
            return pos,vel
    else:
        if returnPhi:
            return pos,phi
        else:
            return pos