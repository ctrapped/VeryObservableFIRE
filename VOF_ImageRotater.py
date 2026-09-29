from scipy.ndimage import rotate
from astropy.io import fits
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import subprocess
import argparse
import os

def RotateAnnotation(hf,phi,npix):
    return rotate(np.reshape(np.array(hf['annotation']),[npix,npix]) , angle=phi,reshape=False)

from VOF_convert_to_fits import convert_to_fits
from VOF_ConvertDataset import AppendToAnnotationsFile

def Denoise(spectra,fov,observer_distance,obs_spatial_res_arcseconds,dnu_kmps,template_fits,sofia_dir,sofia_base_path):
    #Convert to Fits
    fits_filedir = "temp.fits"
    convert_to_fits(spectra,fits_filedir,fov,observer_distance,obs_spatial_res_arcseconds,dnu_kmps,template_fits)

    #Run Or Load SOFIA-2 Mask
    subprocess.run(["bash", "RunSofiaForVOF.sh", sofia_dir, sofia_base_path], check=True)

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


def ConvertSpectraToMomentMaps(spectra,bandwidth_km_s):
    Nspec = np.shape(spectra)[2]
    dv = bandwidth_km_s / Nspec

    velocities = np.linspace(-bandwidth_km_s/2,bandwidth_km_s/2,Nspec)
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

def RotateData(imageDirBase , annotationDirBase, galName, inclination, position_angle, Nsnap, tag, masked, angles=[0,90,180,270],SavePNGs=False,denoise=False,DoTimeAveraging=False,template_fits=None,sofia_dir=None,sofia_base_path=None):
    saveSpectra=True
    fov=observer_distance=obs_spatial_res_arcseconds=dnu_kmps=bandwidth_km_s=None
    try:
        hfImage = h5py.File(imageDirBase+".hdf5",'r')
        spectra = np.array(hfImage['spectra'])
        #Observation parameters saved alongside the datacube by VOF_GenerateSyntheticImage.py
        fov = hfImage.attrs['fov_kpc']
        observer_distance = hfImage.attrs['observer_distance_kpc']
        obs_spatial_res_arcseconds = hfImage.attrs['beam_arcsec']
        dnu_kmps = hfImage.attrs['dnu_kmps']
        bandwidth_km_s = dnu_kmps * np.shape(spectra)[2]
        hfImage.close()
        print("Found image...")
    except:
        #print("Warning!!! No synthetic image for this inclination...")
        spectra = np.zeros((1,1,1))
        saveSpectra=False

    suffix="_i"+str(inclination)+"_pa"+str(position_angle)+masked+"_"+galName+tag

    #CSV manifests the main pipeline (VOF_ConvertDataset.py) writes/appends to for this inclination/position angle,
    #shared across galaxies and snapshots the same way AppendToAnnotationsFile is used there.
    csv_suffix = "_i"+str(inclination)+"_pa"+str(position_angle)+masked
    csv_MF = annotationDirBase+"_MassFlux"+csv_suffix+".csv"
    csv_Mass = annotationDirBase+"_Mass"+csv_suffix+".csv"
    csv_RC = annotationDirBase+"_RC"+csv_suffix+".csv"
    csv_rVel = annotationDirBase+"_rVel"+csv_suffix+".csv"
    csv_sMF = annotationDirBase+"_sMassFlux"+csv_suffix+".csv"

    hfMF = h5py.File(annotationDirBase+"_MassFlux"+suffix+"_MF_"+str(Nsnap)+".hdf5",'r')
    hfMass = h5py.File(annotationDirBase+"_Mass"+suffix+"_Mass_"+str(Nsnap)+".hdf5",'r')
    hfRC = h5py.File(annotationDirBase+"_RC"+suffix+"_RC_"+str(Nsnap)+".hdf5",'r')
    hfrVel = h5py.File(annotationDirBase+"_rVel"+suffix+"_rVel_"+str(Nsnap)+".hdf5",'r')
    hfsMF = h5py.File(annotationDirBase+"_sMassFlux"+suffix+"_sMF_"+str(Nsnap)+".hdf5",'r')

    #Annotation grid resolution, read from the annotation files themselves rather than the (possibly missing) image
    npix = int(round(np.sqrt(np.size(np.array(hfMF['annotation'])))))


    ####################################################

    if denoise:
       sofia_mask = Denoise(spectra,fov,observer_distance,obs_spatial_res_arcseconds,dnu_kmps,template_fits,sofia_dir,sofia_base_path)
    for phi in angles:
        print("saveSpectra=",saveSpectra)
        if phi!=0: spectra_rot = rotate(spectra,angle=phi,reshape=False)
        else: spectra_rot = np.copy(spectra)

        if phi==0: rotString=""
        else: rotString="_r"+str(int(phi))

        dn_tag=""
        rot_mask=None
        if saveSpectra:
          if denoise:
            dn_tag = "_dn"
            rot_mask = rotate(sofia_mask,angle=phi,reshape=False)
            spectra_rot=spectra_rot[rot_mask>0]
            hf_out = h5py.File(imageDirBase+rotString+"_dn.hdf5",'w')
            hf_out.create_dataset('spectra',data=spectra_rot)
            hf_out.create_dataset('moments',data=ConvertSpectraToMomentMaps(spectra_rot,bandwidth_km_s))
            hf_out.close()
            print("Saved moments?")
          elif phi!=0:
            hf_out = h5py.File(imageDirBase+rotString+".hdf5",'w')
            hf_out.create_dataset('spectra',data=spectra_rot)
            hf_out.create_dataset('moments',data=ConvertSpectraToMomentMaps(spectra_rot,bandwidth_km_s))
            hf_out.close()

          spectra_lr = np.flip(spectra_rot,axis=0)
          hf_out = h5py.File(imageDirBase+rotString+dn_tag+"_lr.hdf5",'w')
          hf_out.create_dataset('spectra',data=spectra_lr)
          hf_out.create_dataset('moments',data=ConvertSpectraToMomentMaps(spectra_lr,bandwidth_km_s))
          hf_out.close()





        MF = RotateAnnotation(hfMF,phi,npix)
        Mass = RotateAnnotation(hfMass,phi,npix)
        RC = RotateAnnotation(hfRC,phi,npix)
        rVel = RotateAnnotation(hfrVel,phi,npix)
        sMF = RotateAnnotation(hfsMF,phi,npix)
             
        newImageName_dn = imageDirBase+rotString+"_dn.hdf5"
        newImageName = imageDirBase+rotString+".hdf5"
        
        

        if phi!=0:
            hfMF_o = h5py.File(annotationDirBase+"_MassFlux"+suffix+"_MF_"+str(Nsnap)+rotString+".hdf5",'w')
            hfMass_o = h5py.File(annotationDirBase+"_Mass"+suffix+"_Mass_"+str(Nsnap)+rotString+".hdf5",'w')
            hfRC_o = h5py.File(annotationDirBase+"_RC"+suffix+"_RC_"+str(Nsnap)+rotString+".hdf5",'w')
            hfrVel_o = h5py.File(annotationDirBase+"_rVel_"+suffix+"_rVel_"+str(Nsnap)+rotString+".hdf5",'w')
            hfsMF_o = h5py.File(annotationDirBase+"_sMassFlux_"+suffix+"_sMF_"+str(Nsnap)+rotString+".hdf5",'w')
        
            WriteAnnotation(hfMF_o,MF,newImageName,newImageName_dn)
            WriteAnnotation(hfMass_o,Mass,newImageName,newImageName_dn)
            WriteAnnotation(hfRC_o,RC,newImageName,newImageName_dn)
            WriteAnnotation(hfrVel_o,rVel,newImageName,newImageName_dn)
            WriteAnnotation(hfsMF_o,sMF,newImageName,newImageName_dn)

            AppendToAnnotationsFile(csv_MF,newImageName,MF.flatten())
            AppendToAnnotationsFile(csv_Mass,newImageName,Mass.flatten())
            AppendToAnnotationsFile(csv_RC,newImageName,RC.flatten())
            AppendToAnnotationsFile(csv_rVel,newImageName,rVel.flatten())
            AppendToAnnotationsFile(csv_sMF,newImageName,sMF.flatten())



        hfMF_lr = h5py.File(annotationDirBase+"_MassFlux"+suffix+"_MF_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        hfMass_lr = h5py.File(annotationDirBase+"_Mass"+suffix+"_Mass_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        hfRC_lr = h5py.File(annotationDirBase+"_RC_"+suffix+"_RC_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        hfrVel_lr = h5py.File(annotationDirBase+"_rVel"+suffix+"_rVel_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        hfsMF_lr = h5py.File(annotationDirBase+"_sMassFlux"+suffix+"_sMF_"+str(Nsnap)+rotString+"_lr.hdf5",'w')
        
        newImageName_dn = imageDirBase+rotString+"_dn_lr.hdf5"
        newImageName = imageDirBase+rotString+"_lr.hdf5"
        
        WriteAnnotation(hfMF_lr,np.flip(MF,axis=0),newImageName,newImageName_dn)
        WriteAnnotation(hfMass_lr,np.flip(Mass,axis=0),newImageName,newImageName_dn)
        WriteAnnotation(hfRC_lr,np.flip(RC,axis=0),newImageName,newImageName_dn)
        WriteAnnotation(hfrVel_lr,np.flip(rVel,axis=0),newImageName,newImageName_dn)
        WriteAnnotation(hfsMF_lr,np.flip(sMF,axis=0),newImageName,newImageName_dn)

        AppendToAnnotationsFile(csv_MF,newImageName,np.flip(MF,axis=0).flatten())
        AppendToAnnotationsFile(csv_Mass,newImageName,np.flip(Mass,axis=0).flatten())
        AppendToAnnotationsFile(csv_RC,newImageName,np.flip(RC,axis=0).flatten())
        AppendToAnnotationsFile(csv_rVel,newImageName,np.flip(rVel,axis=0).flatten())
        AppendToAnnotationsFile(csv_sMF,newImageName,np.flip(sMF,axis=0).flatten())

            
            
def _parse_args():
    parser = argparse.ArgumentParser(description="Rotate/augment VOF synthetic images and annotations, optionally denoising via SoFiA-2.")
    parser.add_argument("--data-root", default="/Volumes/wde4tb/simulation_snapshots/fire-2", help="Base directory containing per-galaxy simulation outputs. Each combination is expected at <data-root>/<gal-name>/vof_outputs/i<inclination>/training/.")
    parser.add_argument("--gal-names", nargs="+", default=["m12m","m12i","m12f","m12b"], help="Simulation names to process, e.g. m12m m12i m12f m12b.")
    parser.add_argument("--inclinations", nargs="+", type=int, default=[50,60], help="Inclinations (degrees) to process.")
    parser.add_argument("--position-angles", nargs="+", type=int, default=[0,45,90,135,180,225,270,315], help="Position angles (degrees) to process.")
    parser.add_argument("--snapshots", nargs="+", type=int, default=[600], help="Snapshot numbers to process.")
    parser.add_argument("--tags", nargs="+", default=[""], help="Filename tags to process (default: ['']).")
    parser.add_argument("--masked-tags", nargs="+", default=[""], help="Masked-run filename suffixes to process (default: [''], i.e. unmasked only).")
    parser.add_argument("--denoise", action="store_true", help="Run SoFiA-2 masking via --sofia-dir/--sofia-base-path/--template-fits.")
    parser.add_argument("--sofia-dir", default=".", help="Path to the SoFiA-2 install directory (used when --denoise is set).")
    parser.add_argument("--sofia-base-path", default="./", help="Directory containing the FITS cube passed to SoFiA-2 (used when --denoise is set).")
    parser.add_argument("--template-fits", default="template.fits", help="FITS file to copy the header from when converting to FITS (used when --denoise is set).")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()

    for galName in args.gal_names:
     for inclination in args.inclinations:
      for pa in args.position_angles:
       for Nsnap in args.snapshots:
        for tag in args.tags:
          for masked in args.masked_tags:
                rootDir = os.path.join(args.data_root, galName, "vof_outputs", "i"+str(inclination), "training") + "/"
                imageDirBase = rootDir+galName+tag+"_i"+str(inclination)+"_pa"+str(pa)+"_"+str(Nsnap)+"_image"+masked+"_fullSpectra"
                annotationDirBase = rootDir+"training_annotations"
                RotateData(imageDirBase , annotationDirBase, galName,inclination,pa,Nsnap,tag,masked,denoise=args.denoise,DoTimeAveraging=False,
                           template_fits=args.template_fits,sofia_dir=args.sofia_dir,sofia_base_path=args.sofia_base_path)
                print("Rotated ",imageDirBase)