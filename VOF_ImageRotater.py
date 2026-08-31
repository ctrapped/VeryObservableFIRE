from scipy.ndimage import rotate
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import subprocess

def RotateAnnotation(hf,phi,npix):
    return rotate(np.reshape(np.array(hf['annotation']),[npix,npix]) , angle=phi,reshape=False)
    
#Save these be saved in the image somewhere or read from param file
observer_distance = 10000 #10 Mpc in kpc
fov = 60 #kpc
obs_spatial_res_arcseconds = 6
dnu_kmps = 5.4

from VOF_convert_to_fits import convert_to_fits

def Denoise(spectra):
    #Convert to Fits
    fits_filedir = "temp.fits"
    convert_to_fits(spectra,fits_filedir,fov,observer_distance,obs_spatial_res_arcseconds,dnu_kmps)

    #Run Or Load SOFIA-2 Mask
    subprocess.run(["bash", "RunSofiaForVOF.sh"], check=True)
    
    sofia_filedir = "temp_mask.fits"
    with fits.open(sofia_filedir) as hdul:
        header = hdul[0].header
        mask = hdul[0].data  # This is a NumPy array

    sofia_mask =  np.transpose(mask, (1,2,0))

    return sofia_mask

def WriteDataset(hf,annotation,imageDirBase):
    hf.create_dataset('annotation',data=annotation)
    hf.create_dataset('imageName',data=imageDirBase+"_r0.hdf5")  
    hf.create_dataset('imageName_dn',data=imageDirBase+"_r0_dn.hdf5")
    
def WriteAnnotation(hf,annotation,imageName,imageName_dn):
    hf.create_dataset('annotation',data=annotation.flatten())
    hf.create_dataset('imageName',data=imageName)
    hf.create_dataset('imageName_dn',data=imageName_dn)
    hf.close()


def ConvertSpectraToMomentMaps(spectra):
    f0 = 1420.4 * np.power(10.,6.) # in hz
    bandwidth_km_s = 400
    res_km_s = 5.2
    c_km_s = 3*10**5 #speed of light in km/s
    Nspec = int(np.ceil(bandwidth_km_s / res_km_s))
    bandwidth = f0*c_km_s * (1 / (c_km_s-bandwidth_km_s/2) - 1 / (c_km_s+bandwidth_km_s/2))
    dv = bandwidth / Nspec
    
    velocities = np.linspace(-200,200,np.shape(spectra)[2])    
    moment0 = np.sum(spectra*dv,axis=2)
    moment0[moment0==0]=1e-10
    
    moment1_integrand = np.copy(spectra)*0
    moment2_integrand = np.copy(spectra)*0
    for s in range(0,Nspec):
        moment1_integrand[:,:,s] = spectra[:,:,s] * velocities[s]
    moment1 = np.divide( np.sum( moment1_integrand * dv,axis=2) , moment0 )
    
    for s in range(0,Nspec):
        moment2_integrand[:,:,s] = np.multiply(spectra[:,:,s] , np.power(velocities[s]-moment1[:,:],2))
        
        
    moment2 = np.divide( np.sum( moment2_integrand * dv,axis=2) , moment0 )

    momentMap = np.zeros((np.shape(spectra)[0],np.shape(spectra)[1],3))
    momentMap[:,:,0]=moment0
    momentMap[:,:,1]=moment1
    momentMap[:,:,2]=moment2
    
    return momentMap

def RotateData(imageDirBase , annotationDirBase, galName, inclination, Nsnap, tag, masked, angles=[0,90,180,270],SavePNGs=False,denoiseLevel=0,DoTimeAveraging=False):
    saveSpectra=True
    try:
        hfImage = h5py.File(imageDirBase+".hdf5",'r')
        spectra = np.array(hfImage['spectra'])
        hfImage.close()
        print("Found image...")
    except:
        #print("Warning!!! No synthetic image for this inclination...")
        spectra = np.zeros((40,40,77))
        saveSpectra=False
    
    suffix="_i"+str(inclination)+"_pa"+str(pa)+masked+"_"+galName+tag

    hfMF = h5py.File(annotationDirBase+"_MassFlux"+suffix+"_MF_"+str(Nsnap)+".hdf5",'r')
    hfMass = h5py.File(annotationDirBase+"_Mass"+suffix+"_Mass_"+str(Nsnap)+".hdf5",'r')
    hfRC = h5py.File(annotationDirBase+"_RC"+suffix+"_RC_"+str(Nsnap)+".hdf5",'r')
    hfInc = h5py.File(annotationDirBase+"_inclination"+suffix+"_inc_"+str(Nsnap)+".hdf5",'r')
    hfrVel = h5py.File(annotationDirBase+"_rVel"+suffix+"_rVel_"+str(Nsnap)+".hdf5",'r')
    hfsMF = h5py.File(annotationDirBase+"_sMassFlux"+suffix+"_sMF_"+str(Nsnap)+".hdf5",'r')

    
    ####################################################

    if denoiseLevel>0:
       sofia_mask = Denoise(spectra)
    for phi in angles:
        print("saveSpectra=",saveSpectra)
        if phi!=0: spectra_rot = rotate(spectra,angle=phi,reshape=False)
        else: spectra_rot = np.copy(spectra)

        if phi==0: rotString=""
        else: rotString="_r"+str(int(phi))

        dn_tag=""
        rot_mask=None
        if saveSpectra:
          if denoiseLevel>0:
            dn_tag = "_dn"
            rot_mask = rotate(sofia_mask,angle=phi,reshape=False)
            spectra_rot=spectra_rot[rot_mask>0]
            hf_out = h5py.File(imageDirBase+rotString+"_dn.hdf5",'w')
            hf_out.create_dataset('spectra',data=spectra_rot)
            hf_out.create_dataset('moments',data=ConvertSpectraToMomentMaps(spectra_rot))
            hf_out.close()
            print("Saved moments?")
          elif phi!=0:
            hf_out = h5py.File(imageDirBase+rotString+".hdf5",'w')
            hf_out.create_dataset('spectra',data=spectra_rot)
            hf_out.create_dataset('moments',data=ConvertSpectraToMomentMaps(spectra_rot))
            hf_out.close()
            
          spectra_lr = np.flip(spectra_rot,axis=0)
          hf_out = h5py.File(imageDirBase+rotString+dn_tag+"_lr.hdf5",'w')
          hf_out.create_dataset('spectra',data=spectra_lr)
          hf_out.create_dataset('moments',data=ConvertSpectraToMomentMaps(spectra_lr))
          hf_out.close()
        
          #spectra_ud = np.flip(spectra_rot,axis=1)
          #hf_out = h5py.File(imageDirBase+rotString+dn_tag+"_ud.hdf5",'w')
          #hf_out.create_dataset('spectra',data=spectra_ud)
          #hf_out.create_dataset('moments',data=ConvertSpectraToMomentMaps(spectra_ud))
          #hf_out.close()
        
          #spectra_lr_ud = np.flip(spectra_lr,axis=1)
          #hf_out = h5py.File(imageDirBase+rotString+dn_tag+"_lr_ud.hdf5",'w')
          #hf_out.create_dataset('spectra',data=spectra_lr_ud)
          #hf_out.create_dataset('moments',data=ConvertSpectraToMomentMaps(spectra_lr_ud))
          #hf_out.close()

        
    


        npix,npix,spec = np.shape(spectra)
        MF = RotateAnnotation(hfMF,phi,npix)
        Mass = RotateAnnotation(hfMass,phi,npix)
        RC = RotateAnnotation(hfRC,phi,npix)
        Inc = RotateAnnotation(hfInc,phi,npix)
        rVel = RotateAnnotation(hfrVel,phi,npix)
        sMF = RotateAnnotation(hfsMF,phi,npix)
             
        newImageName_dn = imageDirBase+rotString+"_dn.hdf5"
        newImageName = imageDirBase+rotString+".hdf5"
        
        

        if phi!=0:
            hfMF_o = h5py.File(annotationDirBase+"_MassFlux"+suffix+"_MF_"+str(Nsnap)+rotString+".hdf5",'w')
            hfMass_o = h5py.File(annotationDirBase+"_Mass"+suffix+"_Mass_"+str(Nsnap)+rotString+".hdf5",'w')
            hfRC_o = h5py.File(annotationDirBase+"_RC"+suffix+"_RC_"+str(Nsnap)+rotString+".hdf5",'w')
            hfInc_o = h5py.File(annotationDirBase+"_inclination_"+suffix+"_inc_"+str(Nsnap)+rotString+".hdf5",'w')
            hfrVel_o = h5py.File(annotationDirBase+"_rVel_"+suffix+"_rVel_"+str(Nsnap)+rotString+".hdf5",'w')
            hfsMF_o = h5py.File(annotationDirBase+"_sMassFlux_"+suffix+"_sMF_"+str(Nsnap)+rotString+".hdf5",'w')
        
            WriteAnnotation(hfMF_o,MF,newImageName,newImageName_dn)
            WriteAnnotation(hfMass_o,Mass,newImageName,newImageName_dn)
            WriteAnnotation(hfRC_o,RC,newImageName,newImageName_dn)
            WriteAnnotation(hfInc_o,Inc,newImageName,newImageName_dn)
            WriteAnnotation(hfrVel_o,rVel,newImageName,newImageName_dn)
            WriteAnnotation(hfsMF_o,sMF,newImageName,newImageName_dn)



        hfMF_lr = h5py.File(annotationDirBase+"_MassFlux"+suffix+"_MF_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        hfMass_lr = h5py.File(annotationDirBase+"_Mass"+suffix+"_Mass_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        hfRC_lr = h5py.File(annotationDirBase+"_RC_"+suffix+"_RC_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        hfInc_lr = h5py.File(annotationDirBase+"_inclination"+suffix+"_inc_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        hfrVel_lr = h5py.File(annotationDirBase+"_rVel"+suffix+"_rVel_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        hfsMF_lr = h5py.File(annotationDirBase+"_sMassFlux"+suffix+"_sMF_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        
        newImageName_dn = imageDirBase+rotString+"_dn_lr.hdf5"
        newImageName = imageDirBase+rotString+"_lr.hdf5"
        
        WriteAnnotation(hfMF_lr,np.flip(MF,axis=0),newImageName,newImageName_dn)
        WriteAnnotation(hfMass_lr,np.flip(Mass,axis=0),newImageName,newImageName_dn)
        WriteAnnotation(hfRC_lr,np.flip(RC,axis=0),newImageName,newImageName_dn)
        WriteAnnotation(hfInc_lr,np.flip(Inc,axis=0),newImageName,newImageName_dn)
        WriteAnnotation(hfrVel_lr,np.flip(rVel,axis=0),newImageName,newImageName_dn)
        WriteAnnotation(hfsMF_lr,np.flip(sMF,axis=0),newImageName,newImageName_dn)

        

        #hfMF_ud = h5py.File(annotationDirBase+"_MassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_MF_"+str(Nsnap)+rotString+"_ud.hdf5",'w')
        #hfMass_ud = h5py.File(annotationDirBase+"_Mass_i"+str(inclination)+masked+"_"+galName+tag+"_Mass_"+str(Nsnap)+rotString+"_ud.hdf5",'w')
        #hfRC_ud = h5py.File(annotationDirBase+"_RC_i"+str(inclination)+masked+"_"+galName+tag+"_RC_"+str(Nsnap)+rotString+"_ud.hdf5",'w')
        #hfInc_ud = h5py.File(annotationDirBase+"_inclination_i"+str(inclination)+masked+"_"+galName+tag+"_inc_"+str(Nsnap)+rotString+"_ud.hdf5",'w')
        #hfrVel_ud = h5py.File(annotationDirBase+"_rVel_i"+str(inclination)+masked+"_"+galName+tag+"_rVel_"+str(Nsnap)+rotString+"_ud.hdf5",'w')
        #hfsMF_ud = h5py.File(annotationDirBase+"_sMassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_sMF_"+str(Nsnap)+rotString+"_ud.hdf5",'w')
        
        #newImageName_dn = imageDirBase+rotString+"_dn_ud.hdf5"
        #newImageName = imageDirBase+rotString+"_ud.hdf5"
        
        #WriteAnnotation(hfMF_ud,np.flip(MF,axis=1),newImageName,newImageName_dn)
        #WriteAnnotation(hfMass_ud,np.flip(Mass,axis=1),newImageName,newImageName_dn)
        #WriteAnnotation(hfRC_ud,np.flip(RC,axis=1),newImageName,newImageName_dn)
        #WriteAnnotation(hfInc_ud,np.flip(Inc,axis=1),newImageName,newImageName_dn)
        #WriteAnnotation(hfrVel_ud,np.flip(rVel,axis=1),newImageName,newImageName_dn)
        #WriteAnnotation(hfsMF_ud,np.flip(sMF,axis=1),newImageName,newImageName_dn)
        
        

        #hfMF_lr_ud = h5py.File(annotationDirBase+"_MassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_MF_"+str(Nsnap)+rotString+"_lr_ud.hdf5",'w')
        #hfMass_lr_ud = h5py.File(annotationDirBase+"_Mass_i"+str(inclination)+masked+"_"+galName+tag+"_Mass_"+str(Nsnap)+rotString+"_lr_ud.hdf5",'w')
        #hfRC_lr_ud = h5py.File(annotationDirBase+"_RC_i"+str(inclination)+masked+"_"+galName+tag+"_RC_"+str(Nsnap)+rotString+"_lr_ud.hdf5",'w')
        #hfInc_lr_ud = h5py.File(annotationDirBase+"_inclination_i"+str(inclination)+masked+"_"+galName+tag+"_inc_"+str(Nsnap)+rotString+"_lr_ud.hdf5",'w')
        #hfrVel_lr_ud = h5py.File(annotationDirBase+"_rVel_i"+str(inclination)+masked+"_"+galName+tag+"_rVel_"+str(Nsnap)+rotString+"_lr_ud.hdf5",'w')
        #hfsMF_lr_ud = h5py.File(annotationDirBase+"_sMassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_sMF_"+str(Nsnap)+rotString+"_lr_ud.hdf5",'w')
        
        #newImageName_dn = imageDirBase+rotString+"_dn_lr_ud.hdf5"
        #newImageName = imageDirBase+rotString+"_lr_ud.hdf5"
        
        #WriteAnnotation(hfMF_lr_ud,np.flip(np.flip(MF,axis=0),axis=1),newImageName,newImageName_dn)
        #WriteAnnotation(hfMass_lr_ud,np.flip(np.flip(Mass,axis=0),axis=1),newImageName,newImageName_dn)
        #WriteAnnotation(hfRC_lr_ud,np.flip(np.flip(RC,axis=0),axis=1),newImageName,newImageName_dn)
        #WriteAnnotation(hfInc_lr_ud,np.flip(np.flip(Inc,axis=0),axis=1),newImageName,newImageName_dn)
        #WriteAnnotation(hfrVel_lr_ud,np.flip(np.flip(rVel,axis=0),axis=1),newImageName,newImageName_dn)
        #WriteAnnotation(hfsMF_lr_ud,np.flip(np.flip(sMF,axis=0),axis=1),newImageName,newImageName_dn)
        
        

            
            
#galNames =['m12m','m12i','m12f','m12b','m12c','m12r','m12z','m12w','m12_elvis_RomeoJuliet','m12_elvis_ThelmaLouise','m12_elvis_RomulusRemus']
galNames=['m12m','m12i','m12f','m12b']
denoiseLevel=0
for galName in galNames:
 for inclination in [50,60]:
  for pa in [0,45,90,135,180,225,270,315]:
   for Nsnap in [600]:
    for tag in ['']:
      for masked in ['']:
        #try:
        if True:
            rootDir="../CoNNGaFit/galfitData2/fire2_03112024/i"+str(inclination)+"/training/"
            rootDir="/Volumes/wde4tb/simulation_snapshots/fire-2/"+galName+"/vof_outputs/i"+str(inclination)+"/training/"
            imageDirBase = rootDir+galName+tag+"_cr700_i"+str(inclination)+"_pa"+str(pa)+"_"+str(Nsnap)+"_image_04172023"+masked+"_fullSpectra"
            annotationDirBase = rootDir+"training_annotations"
            RotateData(imageDirBase , annotationDirBase, galName,inclination,Nsnap,tag,masked,denoiseLevel=denoiseLevel,DoTimeAveraging=False)  
            print("Rotated ",imageDirBase)      
       # except:
        #else:
         #   imageDirBase = rootDir+galName+tag+"_cr700_i"+str(inclination)+"_pa"+str(pa)+"_"+str(Nsnap)+"_image_04172023"+masked+"_fullSpectra"
       #     print("Warning: could not rotate ",imageDirBase)      