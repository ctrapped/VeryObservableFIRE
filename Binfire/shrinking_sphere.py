import numpy as np
import h5py as h5py
import os.path
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import math
import sys
#Taken in part from pynbody.analysis.halo

def remove_vals(the_list,val):
  return [value for value in the_list if value!=val]

def center_of_mass_velocity(Smas,Svx,Svy,Svz):
   #mtot = np.sum(DMmas) + np.sum(Smas) #Total mass
   mtot = np.sum(Smas) #Total mass
   v_com=np.zeros((3))
   print(np.shape(Svx),np.shape(Svy),np.shape(Svz),np.shape(Smas))
   v_com[:] = np.divide(np.dot(np.array([Svx,Svy,Svz]),Smas), mtot) #Mass weighted vel. avg #TRY JUST DOING Stars
   #v_com[:] = np.divide(np.add(np.dot(np.array([DMvx,DMvy,DMvz]),DMmas),np.dot(np.array([Svx,Svy,Svz]),Smas)), mtot) #Mass weighted vel. avg #TRY JUST DOING Stars
   return v_com

def center_of_mass_vel_gen(mas,vel):
    mtot = np.sum(mas)
    v_com = np.zeros((3))
    print("Shape of mass and vel vectors:",np.shape(vel),np.shape(mas))
    v_com[0] = np.dot(mas,vel[:,0])/mtot
    v_com[1] = np.dot(mas,vel[:,1])/mtot
    v_com[2] = np.dot(mas,vel[:,2])/mtot
    return v_com



def center_of_mass_vel_MC(vel,rhat,step,vel_center=[0,0,0]):
    #start with very basic MC
    N = np.shape(vel)[0]
    #vel_center = np.zeros((3))
    vel_rmag = np.zeros((N))
    vel_r = np.zeros((N,3))
    itr_count = np.zeros((3))
    min_step = 0.001
    max_iter = 10
    for dim in [1,0,2]:
        vel_rmag[:] = np.multiply(vel[:,0],rhat[:,0])+np.multiply(vel[:,1],rhat[:,1])+np.multiply(vel[:,2],rhat[:,2])
        vel_r[:,0] = np.multiply(rhat[:,0],vel_rmag)
        vel_r[:,1] = np.multiply(rhat[:,1],vel_rmag)
        vel_r[:,2] = np.multiply(rhat[:,2],vel_rmag)
         
        x_dispersion = np.std(vel_r[:,0])
        y_dispersion = np.std(vel_r[:,1])
        z_dispersion = np.std(vel_r[:,2])

        current_min = x_dispersion**2+y_dispersion**2+z_dispersion**2
        print("ANALYZING DIMENSION",dim)
        print("Initial dispersion:",current_min)

        for ii in range(0,max_iter):
          vel[:,dim]=vel[:,dim]-step
          vel_rmag[:] = np.multiply(vel[:,0],rhat[:,0])+np.multiply(vel[:,1],rhat[:,1])+np.multiply(vel[:,2],rhat[:,2])
          vel_r[:,0] = np.multiply(rhat[:,0],vel_rmag)
          vel_r[:,1] = np.multiply(rhat[:,1],vel_rmag)
          vel_r[:,2] = np.multiply(rhat[:,2],vel_rmag)
         
          x_dispersion = np.std(vel_r[:,0])
          y_dispersion = np.std(vel_r[:,1])
          z_dispersion = np.std(vel_r[:,2])
          disp2_subtraction = x_dispersion**2+y_dispersion**2+z_dispersion**2
          print("Dispersion from subtraction:",disp2_subtraction)

          vel[:,dim]=vel[:,dim]+2*step #net is just +step
          vel_rmag[:] = np.multiply(vel[:,0],rhat[:,0])+np.multiply(vel[:,1],rhat[:,1])+np.multiply(vel[:,2],rhat[:,2])
          vel_r[:,0] = np.multiply(rhat[:,0],vel_rmag)
          vel_r[:,1] = np.multiply(rhat[:,1],vel_rmag)
          vel_r[:,2] = np.multiply(rhat[:,2],vel_rmag)
         
          x_dispersion = np.std(vel_r[:,0])
          y_dispersion = np.std(vel_r[:,1])
          z_dispersion = np.std(vel_r[:,2])
          disp2_addition = x_dispersion**2+y_dispersion**2+z_dispersion**2
          print("Dispersion from addition:",disp2_addition)
     
          if ((disp2_subtraction < current_min) or (disp2_addition < current_min)):
              if disp2_subtraction<disp2_addition:
                  vel[:,dim]=vel[:,dim]-2*step #return to the first step
                  vel_center[dim]+=step
                  current_min=disp2_subtraction
                  print("Choosing Subtraction. Vel_center is:",vel_center)
              else:
                  vel_center[dim]-=step
                  current_min=disp2_addition
                  print("Choosing Addition. Vel_center is:",vel_center)
          else:
              vel[:,dim] = vel[:,dim]-step #undo last step
              itr_count[dim] = ii
              print("Finished iterating along this dimension. Vel_center is:",vel_center)
              break; #exit loop, assuming no local minima
    print("step,step/10,min_step are:",step,step/10,min_step)
    if ((float(step)/10.)>min_step):
        print("Iterating at finer resolution. Step =",step/10)
        vel_center = center_of_mass_vel_MC(vel,rhat,float(step)/10.,vel_center)

    return vel_center

    





def center_of_mass(Smas,r_s): #Only Baryonic matter
   #mtot = np.sum(Gmas) + np.sum(Smas) #Total mass
   pos_com=np.zeros((3))
   #print(np.shape(np.transpose(r_s)),' ',np.shape(Smas)
   pos_com[:] = np.divide(  np.dot(np.transpose(r_s),Smas) , np.sum(Smas)  )
   #pos_com[:] = np.divide(np.add(np.dot(np.array([Gx,Gy,Gz]),Gmas),np.dot(np.array([Sx,Sy,Sz]),Smas)),mtot) #Try only using star particles
   print("New center of mass:",pos_com)
   return pos_com
  

def shrink_sphere(Smas,Spos,wiggle_count,r=None,rmin=10,shrink_factor=0.7,old_com = None):
    delta_cutoff = 0.000001
    NS = np.size(Smas)
    print(NS," Stars in sphere.")
   
    print("Calculating new center of mass...")
    pos_com = center_of_mass(Smas,Spos)

    #Srel = np.subtract(Spos,pos_com)
    #print(np.shape(Srel)
    r2_S = np.add(np.power(Spos[:,0]-pos_com[0],2),np.add(np.power(Spos[:,1]-pos_com[1],2),np.power(Spos[:,2]-pos_com[2],2)))
    #r2_S = np.add(np.power(np.subtract(Sx,pos_com[0]),2),np.add(np.power(np.subtract(Sy,pos_com[1]),2),np.power(np.subtract(Sz,pos_com[2]),2)))
    #r_S = abs(np.linalg.norm(Srel))

    print(np.shape(r2_S))
    r_new = r*shrink_factor 
    print("Rnew is",r_new  )
   
    #print(np.where(r2_S<np.power(r,2))[0]
    #print(Smas[r2_S<np.power(r,2)]
    #print("# of particles in new sphere: ",NS2
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
        



























def shrink_sphere_parallel(Smas,Spos,wiggle_count,r=None,rMin=10,shrink_factor=0.7,old_com = None):
    delta_cutoff = 0.000001
    NS = np.size(Smas)
    print(NS," Stars in sphere.")
   
    print("Calculating new center of mass...")
    pos_com = center_of_mass(Smas,Spos)


    while rNew > rMin:
        r2_S = np.add(np.power(Spos[:,0]-pos_com[0],2),np.add(np.power(Spos[:,1]-pos_com[1],2),np.power(Spos[:,2]-pos_com[2],2)))
        rNew = r*shrink_factor 
        print("rNew is:",rNew  )
        Sidx = np.where( r2_S < rNew*rNew )


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
        



