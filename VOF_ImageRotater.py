from scipy.ndimage import rotate
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def RotateAnnotation(hf,phi,npix):
    return rotate(np.reshape(np.array(hf['annotation']),[npix,npix]) , angle=phi,reshape=False)
    
def Denoise(spectra,denoiseLevel):
    npix,npix,nspec = np.shape(spectra)
    sq=np.multiply(spectra[0:2,0:2,:],spectra[0:2,0:2,:]) + np.multiply(spectra[0:2,npix-3:npix-1,:],spectra[0:2,npix-3:npix-1,:]) + np.multiply(spectra[npix-3:npix-1,npix-3:npix-1,:],spectra[npix-3:npix-1,npix-3:npix-1,:])+ np.multiply(spectra[npix-3:npix-1,0:2,:],spectra[npix-3:npix-1,0:2,:])

    sigma = np.sqrt(np.mean(sq/4))
    spectra_dn = np.copy(spectra)
    spectra_dn[spectra_dn < sigma*denoiseLevel] = 0
    return spectra_dn

def WriteDataset(hf,annotation,imageDirBase):
    hf.create_dataset('annotation',data=annotation)
    hf.create_dataset('imageName',data=imageDirBase+"_r0.hdf5")  
    hf.create_dataset('imageName_dn',data=imageDirBase+"_r0_dn.hdf5")
    
def WriteTimeAverage(hf0,hf1,hf2,hf_out,imageDirBase):
    annotation = np.add(hf0['annotation'], np.add(hf1['annotation'],hf2['annotation'])) / 3.
    hf_out.create_dataset('annotation',data=annotation)
    newImageName = imageDirBase+".hdf5"
    newImageName_dn = imageDirBase+"_dn.hdf5"
    hf_out.create_dataset('imageName',data=newImageName)
    hf_out.create_dataset('imageName_dn',data=newImageName_dn)
    
def WriteAnnotation(hf,annotation,imageName,imageName_dn):
    hf.create_dataset('annotation',data=annotation.flatten())
    hf.create_dataset('imageName',data=imageName)
    hf.create_dataset('imageName_dn',data=imageName_dn)
    hf.close()


def RotateData(imageDirBase , annotationDirBase, galName, inclination, Nsnap, tag, masked, angles=[0,90,180,270],SavePNGs=False,denoiseLevel=0,DoTimeAveraging=False):
    hfImage = h5py.File(imageDirBase+".hdf5",'r')
    spectra = np.array(hfImage['spectra'])
    hfImage.close()
    
    
    hfMF = h5py.File(annotationDirBase+"_MassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_MF_"+str(Nsnap)+".hdf5",'r')
    hfMass = h5py.File(annotationDirBase+"_Mass_i"+str(inclination)+masked+"_"+galName+tag+"_Mass_"+str(Nsnap)+".hdf5",'r')
    hfRC = h5py.File(annotationDirBase+"_RC_i"+str(inclination)+masked+"_"+galName+tag+"_RC_"+str(Nsnap)+".hdf5",'r')
    hfInc = h5py.File(annotationDirBase+"_inclination_i"+str(inclination)+masked+"_"+galName+tag+"_inc_"+str(Nsnap)+".hdf5",'r')
    hfrVel = h5py.File(annotationDirBase+"_rVel_i"+str(inclination)+masked+"_"+galName+tag+"_rVel_"+str(Nsnap)+".hdf5",'r')
    hfsMF = h5py.File(annotationDirBase+"_sMassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_sMF_"+str(Nsnap)+".hdf5",'r')
    
    if DoTimeAveraging:
        try:
            hfMF_m1 = h5py.File(annotationDirBase+"_MassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_MF_"+str(Nsnap-1)+".hdf5",'r')
            hfMF_m2 = h5py.File(annotationDirBase+"_MassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_MF_"+str(Nsnap-2)+".hdf5",'r')
            hfMF_o =  h5py.File(annotationDirBase+"_MassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_MF_"+str(Nsnap)+"_timeAvg.hdf5",'w')
            WriteTimeAverage(hfMF,hfMF_m1,hfMF_m2,hfMF_o,imageDirBase)
            hfMF_o.close()
            
            hfsMF_m1 = h5py.File(annotationDirBase+"_sMassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_sMF_"+str(Nsnap-1)+".hdf5",'r')
            hfsMF_m2 = h5py.File(annotationDirBase+"_sMassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_sMF_"+str(Nsnap-2)+".hdf5",'r')
            hfsMF_o =  h5py.File(annotationDirBase+"_sMassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_sMF_"+str(Nsnap)+"_timeAvg.hdf5",'w')
            WriteTimeAverage(hfsMF,hfsMF_m1,hfsMF_m2,hfsMF_o,imageDirBase)
            hfMF_o.close()
            
            hfrVel_m1 = h5py.File(annotationDirBase+"_rVel_i"+str(inclination)+masked+"_"+galName+tag+"_rVel_"+str(Nsnap-1)+".hdf5",'r')
            hfrVel_m2 = h5py.File(annotationDirBase+"_rVel_i"+str(inclination)+masked+"_"+galName+tag+"_rVel_"+str(Nsnap-2)+".hdf5",'r')
            hfrVel_o =  h5py.File(annotationDirBase+"_rVel_i"+str(inclination)+masked+"_"+galName+tag+"_rVel_"+str(Nsnap)+"_timeAvg.hdf5",'w')
            WriteTimeAverage(hfrVel,hfrVel_m1,hfrVel_m2,hfrVel_o,imageDirBase)
            hfrVel_o.close()
            
            
            
        except:
            print("Could not do time averaging for",imageDirBase)

    
    ####################################################

    
    for phi in angles:
        if phi!=0: spectra_rot = rotate(spectra,angle=phi,reshape=False)
        else: spectra_rot = np.copy(spectra)

        if phi==0: rotString=""
        else: rotString="_r"+str(int(phi))

        dn_tag=""
        if denoiseLevel>0:
            dn_tag = "_dn"
            spectra_rot = Denoise(spectra_rot,denoiseLevel)
            hf_out = h5py.File(imageDirBase+rotString+"_dn.hdf5",'w')
            hf_out.create_dataset('spectra',data=spectra_rot)
            hf_out.close()
        elif phi!=0:
            hf_out = h5py.File(imageDirBase+rotString+".hdf5",'w')
            hf_out.create_dataset('spectra',data=spectra_rot)
            hf_out.close()
            
        spectra_lr = np.flip(spectra_rot,axis=0)
        hf_out = h5py.File(imageDirBase+rotString+dn_tag+"_lr.hdf5",'w')
        hf_out.create_dataset('spectra',data=spectra_lr)
        hf_out.close()
        
        spectra_ud = np.flip(spectra_rot,axis=1)
        hf_out = h5py.File(imageDirBase+rotString+dn_tag+"_ud.hdf5",'w')
        hf_out.create_dataset('spectra',data=spectra_ud)
        hf_out.close()
        
        spectra_lr_ud = np.flip(spectra_lr,axis=1)
        hf_out = h5py.File(imageDirBase+rotString+dn_tag+"_lr_ud.hdf5",'w')
        hf_out.create_dataset('spectra',data=spectra_lr_ud)
        hf_out.close()

        
    


        npix,npix,spec = np.shape(spectra)
        MF = RotateAnnotation(hfMF,phi,npix)
        Mass = RotateAnnotation(hfMass,phi,npix)
        RC = RotateAnnotation(hfRC,phi,npix)
        Inc = RotateAnnotation(hfInc,phi,npix)
        rVel = RotateAnnotation(hfrVel,phi,npix)
        sMF = RotateAnnotation(hfsMF,phi,npix)
        
        if DoTimeAveraging:
            try:
                newImageName = imageDirBase+rotString+".hdf5"
                newImageName_dn = imageDirBase+rotString+"_dn.hdf5"
            
                MF_m1 = RotateAnnotation(hfMF_m1,phi,npix)
                MF_m2 = RotateAnnotation(hfMF_m2,phi,npix)
                MF_timeAveraged = np.add(MF, np.add(MF_m1,MF_m2)) / 3.
                hfMF_o_ta = h5py.File(annotationDirBase+"_MassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_MF_"+str(Nsnap)+rotString+"_timeAvg.hdf5",'w')
                hfMF_o_ta.create_dataset('annotation',data=MF_timeAveraged.flatten())
                hfMF_o_ta.create_dataset('imageName',data=newImageName)
                hfMF_o_ta.create_dataset('imageName_dn',data=newImageName_dn)
                hfMF_o_ta.close()
                
                sMF_m1 = RotateAnnotation(hfsMF_m1,phi,npix)
                sMF_m2 = RotateAnnotation(hfsMF_m2,phi,npix)
                sMF_timeAveraged = np.add(sMF, np.add(sMF_m1,sMF_m2)) / 3.
                hfsMF_o_ta = h5py.File(annotationDirBase+"_sMassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_sMF_"+str(Nsnap)+rotString+"_timeAvg.hdf5",'w')
                hfsMF_o_ta.create_dataset('annotation',data=sMF_timeAveraged.flatten())
                hfsMF_o_ta.create_dataset('imageName',data=newImageName)
                hfsMF_o_ta.create_dataset('imageName_dn',data=newImageName_dn)
                hfsMF_o_ta.close()
                
                rVel_m1 = RotateAnnotation(hfrVel_m1,phi,npix)
                rVel_m2 = RotateAnnotation(hfrVel_m2,phi,npix)
                rVel_timeAveraged = np.add(rVel, np.add(rVel_m1,rVel_m2)) / 3.
                hfrVel_o_ta = h5py.File(annotationDirBase+"_rVel_i"+str(inclination)+masked+"_"+galName+tag+"_rVel_"+str(Nsnap)+rotString+"_timeAvg.hdf5",'w')
                hfrVel_o_ta.create_dataset('annotation',data=rVel_timeAveraged.flatten())
                hfrVel_o_ta.create_dataset('imageName',data=newImageName)
                hfrVel_o_ta.create_dataset('imageName_dn',data=newImageName_dn)
                hfrVel_o_ta.close()
                

                
            except:
                x=1
             
        newImageName_dn = imageDirBase+rotString+"_dn.hdf5"
        newImageName = imageDirBase+rotString+".hdf5"
        
        if phi!=0:
            hfMF_o = h5py.File(annotationDirBase+"_MassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_MF_"+str(Nsnap)+rotString+".hdf5",'w')
            hfMass_o = h5py.File(annotationDirBase+"_Mass_i"+str(inclination)+masked+"_"+galName+tag+"_Mass_"+str(Nsnap)+rotString+".hdf5",'w')
            hfRC_o = h5py.File(annotationDirBase+"_RC_i"+str(inclination)+masked+"_"+galName+tag+"_RC_"+str(Nsnap)+rotString+".hdf5",'w')
            hfInc_o = h5py.File(annotationDirBase+"_inclination_i"+str(inclination)+masked+"_"+galName+tag+"_inc_"+str(Nsnap)+rotString+".hdf5",'w')
            hfrVel_o = h5py.File(annotationDirBase+"_rVel_i"+str(inclination)+masked+"_"+galName+tag+"_rVel_"+str(Nsnap)+rotString+".hdf5",'w')
            hfsMF_o = h5py.File(annotationDirBase+"_sMassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_sMF_"+str(Nsnap)+rotString+".hdf5",'w')
        
            WriteAnnotation(hfMF_o,MF,newImageName,newImageName_dn)
            WriteAnnotation(hfMass_o,Mass,newImageName,newImageName_dn)
            WriteAnnotation(hfRC_o,RC,newImageName,newImageName_dn)
            WriteAnnotation(hfInc_o,Inc,newImageName,newImageName_dn)
            WriteAnnotation(hfrVel_o,rVel,newImageName,newImageName_dn)
            WriteAnnotation(hfsMF_o,sMF,newImageName,newImageName_dn)



        hfMF_lr = h5py.File(annotationDirBase+"_MassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_MF_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        hfMass_lr = h5py.File(annotationDirBase+"_Mass_i"+str(inclination)+masked+"_"+galName+tag+"_Mass_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        hfRC_lr = h5py.File(annotationDirBase+"_RC_i"+str(inclination)+masked+"_"+galName+tag+"_RC_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        hfInc_lr = h5py.File(annotationDirBase+"_inclination_i"+str(inclination)+masked+"_"+galName+tag+"_inc_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        hfrVel_lr = h5py.File(annotationDirBase+"_rVel_i"+str(inclination)+masked+"_"+galName+tag+"_rVel_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        hfsMF_lr = h5py.File(annotationDirBase+"_sMassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_sMF_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        
        newImageName_dn = imageDirBase+rotString+"_dn_lr.hdf5"
        newImageName = imageDirBase+rotString+"_lr.hdf5"
        
        WriteAnnotation(hfMF_lr,np.flip(MF,axis=0),newImageName,newImageName_dn)
        WriteAnnotation(hfMass_lr,np.flip(Mass,axis=0),newImageName,newImageName_dn)
        WriteAnnotation(hfRC_lr,np.flip(RC,axis=0),newImageName,newImageName_dn)
        WriteAnnotation(hfInc_lr,np.flip(Inc,axis=0),newImageName,newImageName_dn)
        WriteAnnotation(hfrVel_lr,np.flip(rVel,axis=0),newImageName,newImageName_dn)
        WriteAnnotation(hfsMF_lr,np.flip(sMF,axis=0),newImageName,newImageName_dn)

        

        hfMF_ud = h5py.File(annotationDirBase+"_MassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_MF_"+str(Nsnap)+rotString+"_ud.hdf5",'w')
        hfMass_ud = h5py.File(annotationDirBase+"_Mass_i"+str(inclination)+masked+"_"+galName+tag+"_Mass_"+str(Nsnap)+rotString+"_ud.hdf5",'w')
        hfRC_ud = h5py.File(annotationDirBase+"_RC_i"+str(inclination)+masked+"_"+galName+tag+"_RC_"+str(Nsnap)+rotString+"_ud.hdf5",'w')
        hfInc_ud = h5py.File(annotationDirBase+"_inclination_i"+str(inclination)+masked+"_"+galName+tag+"_inc_"+str(Nsnap)+rotString+"_ud.hdf5",'w')
        hfrVel_ud = h5py.File(annotationDirBase+"_rVel_i"+str(inclination)+masked+"_"+galName+tag+"_rVel_"+str(Nsnap)+rotString+"_ud.hdf5",'w')
        hfsMF_ud = h5py.File(annotationDirBase+"_sMassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_sMF_"+str(Nsnap)+rotString+"_ud.hdf5",'w')
        
        newImageName_dn = imageDirBase+rotString+"_dn_ud.hdf5"
        newImageName = imageDirBase+rotString+"_ud.hdf5"
        
        WriteAnnotation(hfMF_ud,np.flip(MF,axis=1),newImageName,newImageName_dn)
        WriteAnnotation(hfMass_ud,np.flip(Mass,axis=1),newImageName,newImageName_dn)
        WriteAnnotation(hfRC_ud,np.flip(RC,axis=1),newImageName,newImageName_dn)
        WriteAnnotation(hfInc_ud,np.flip(Inc,axis=1),newImageName,newImageName_dn)
        WriteAnnotation(hfrVel_ud,np.flip(rVel,axis=1),newImageName,newImageName_dn)
        WriteAnnotation(hfsMF_ud,np.flip(sMF,axis=1),newImageName,newImageName_dn)
        
        

        hfMF_lr_ud = h5py.File(annotationDirBase+"_MassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_MF_"+str(Nsnap)+rotString+"_lr_ud.hdf5",'w')
        hfMass_lr_ud = h5py.File(annotationDirBase+"_Mass_i"+str(inclination)+masked+"_"+galName+tag+"_Mass_"+str(Nsnap)+rotString+"_lr_ud.hdf5",'w')
        hfRC_lr_ud = h5py.File(annotationDirBase+"_RC_i"+str(inclination)+masked+"_"+galName+tag+"_RC_"+str(Nsnap)+rotString+"_lr_ud.hdf5",'w')
        hfInc_lr_ud = h5py.File(annotationDirBase+"_inclination_i"+str(inclination)+masked+"_"+galName+tag+"_inc_"+str(Nsnap)+rotString+"_lr_ud.hdf5",'w')
        hfrVel_lr_ud = h5py.File(annotationDirBase+"_rVel_i"+str(inclination)+masked+"_"+galName+tag+"_rVel_"+str(Nsnap)+rotString+"_lr_ud.hdf5",'w')
        hfsMF_lr_ud = h5py.File(annotationDirBase+"_sMassFlux_i"+str(inclination)+masked+"_"+galName+tag+"_sMF_"+str(Nsnap)+rotString+"_lr_ud.hdf5",'w')
        
        newImageName_dn = imageDirBase+rotString+"_dn_lr_ud.hdf5"
        newImageName = imageDirBase+rotString+"_lr_ud.hdf5"
        
        WriteAnnotation(hfMF_lr_ud,np.flip(np.flip(MF,axis=0),axis=1),newImageName,newImageName_dn)
        WriteAnnotation(hfMass_lr_ud,np.flip(np.flip(Mass,axis=0),axis=1),newImageName,newImageName_dn)
        WriteAnnotation(hfRC_lr_ud,np.flip(np.flip(RC,axis=0),axis=1),newImageName,newImageName_dn)
        WriteAnnotation(hfInc_lr_ud,np.flip(np.flip(Inc,axis=0),axis=1),newImageName,newImageName_dn)
        WriteAnnotation(hfrVel_lr_ud,np.flip(np.flip(rVel,axis=0),axis=1),newImageName,newImageName_dn)
        WriteAnnotation(hfsMF_lr_ud,np.flip(np.flip(sMF,axis=0),axis=1),newImageName,newImageName_dn)
        
        

            
            
galNames =['m12m','m12i','m12f','m12b','m12c','m12r','m12z','m12w','m12_elvis_RomeoJuliet','m12_elvis_ThelmaLouise','m12_elvis_RomulusRemus']
#galNames=['m12_elvis_RomeoJuliet']
denoiseLevel=0

for galName in galNames:
 for inclination in [35,40,45,50,55]:
  for Nsnap in range(581,591):
    for tag in ['','_md']:
      for masked in ['','_masked']:
        try:
        #if True:
            rootDir="../CoNNGaFit/galfitData2/fire2_03112024/i"+str(inclination)+"/training/"
            imageDirBase = rootDir+galName+tag+"_cr700_i"+str(inclination)+"_"+str(Nsnap)+"_image_04172023"+masked+"_fullSpectra"
            annotationDirBase = rootDir+"training_annotations"
            print("Rotating ",imageDirBase)      
            RotateData(imageDirBase , annotationDirBase, galName,inclination,Nsnap,tag,masked,denoiseLevel=denoiseLevel,DoTimeAveraging=False)  
            

        except:
        #else:
            imageDirBase = rootDir+galName+tag+"_cr700_i"+str(inclination)+"_"+str(Nsnap)+"_image_04172023"+masked+"_fullSpectra"
            print("Warning: could not rotate ",imageDirBase)      