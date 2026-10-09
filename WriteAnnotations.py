import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

####Writes the Binfire-binned annotation maps (mass flux, mass, rotation curve, radial velocity) for a single
####inclination/position_angle to disk as .hdf5 (+ optional .png previews), for use as NN training annotations.
####
####Written By Cameron Trapp (ctrapped@gmail.com)

def WriteAnnotations(output, galName, inclination, position_angle, Nsnap, outputSuffix,
                      binnedMass, binnedRadialMassFlux, binnedPhiMassFlux, binnedCylRadMassFlux, binnedCylRadMassFluxCurve,
                      writeRadialVelocity=True,
                      savePNG=False):

    angle_str = "i"+str(inclination)+"_pa"+str(position_angle)
    #Matches the datacube filename GenerateSyntheticImage.py actually writes in MakeDataset's createImages loop (image_name+"_fullSpectra.hdf5" there)
    image_name=output+"i"+str(inclination)+"/"+galName+"_"+angle_str+"_"+str(Nsnap)+"_image"+outputSuffix+"_fullSpectra.hdf5"
    annotationFileDir_MF = output+"i"+str(inclination)+"/training_annotations_MassFlux_"+angle_str+outputSuffix
    annotationFileDir_Mass = output+"i"+str(inclination)+"/training_annotations_Mass_"+angle_str+outputSuffix
    annotationFileDir_RC = output+"i"+str(inclination)+"/training_annotations_RC_"+angle_str+outputSuffix
    annotationFileDir_rVel = output+"i"+str(inclination)+"/training_annotations_rVel_"+angle_str+outputSuffix
    annotationFileDir_sMF = output+"i"+str(inclination)+"/training_annotations_sMassFlux_"+angle_str+outputSuffix
    annotationFileDir_sMF1d = output+"i"+str(inclination)+"/training_annotations_sMassFluxCurve_"+angle_str+outputSuffix

    if savePNG:
        vmax = np.max([-np.min(binnedRadialMassFlux) , np.max(binnedRadialMassFlux)])*.5
        vmin=-vmax
        plt.figure()
        plt.imshow(binnedRadialMassFlux,vmin=vmin,vmax=vmax,cmap='seismic')
        plt.colorbar()
        plt.savefig(annotationFileDir_MF+"_"+galName+"_MF_"+str(Nsnap)+".png")
        plt.close()

        vmax = np.max([-np.min(binnedCylRadMassFlux) , np.max(binnedCylRadMassFlux)])*.5
        vmin=-vmax
        plt.figure()
        plt.imshow(binnedCylRadMassFlux,vmin=vmin,vmax=vmax,cmap='seismic')
        plt.colorbar()
        plt.savefig(annotationFileDir_sMF+"_"+galName+"_sMF_"+str(Nsnap)+".png")
        plt.close()

    hfMF=h5py.File(annotationFileDir_MF+"_"+galName+"_MF_"+str(Nsnap)+".hdf5",'w')
    hfMF.create_dataset('imageName',data=image_name)
    hfMF.create_dataset('annotation',data=binnedRadialMassFlux.flatten())
    hfMF.close()

    hfsMF=h5py.File(annotationFileDir_sMF+"_"+galName+"_sMF_"+str(Nsnap)+".hdf5",'w')
    hfsMF.create_dataset('imageName',data=image_name)
    hfsMF.create_dataset('annotation',data=binnedCylRadMassFlux.flatten())
    hfsMF.close()

    hfsMF1d=h5py.File(annotationFileDir_sMF1d+"_"+galName+"_sMF1d_"+str(Nsnap)+".hdf5",'w')
    hfsMF1d.create_dataset('imageName',data=image_name)
    hfsMF1d.create_dataset('annotation',data=binnedCylRadMassFluxCurve.flatten())
    hfsMF1d.close()

    if writeRadialVelocity:
        binnedMass[binnedMass==0]=1e-10
        if savePNG:
            vmax = np.max([-np.min(np.divide(binnedCylRadMassFlux,binnedMass)) , np.max(np.divide(binnedCylRadMassFlux,binnedMass))])
            vmin=-vmax
            plt.figure()
            plt.imshow(np.divide(binnedCylRadMassFlux,binnedMass),vmin=vmin,vmax=vmax,cmap='seismic')
            plt.colorbar()
            plt.savefig(annotationFileDir_rVel+"_"+galName+"_rVel_"+str(Nsnap)+".png")
            plt.close()

        hfrVel=h5py.File(annotationFileDir_rVel+"_"+galName+"_rVel_"+str(Nsnap)+".hdf5",'w')
        hfrVel.create_dataset('imageName',data=image_name)
        hfrVel.create_dataset('annotation',data=np.divide(binnedCylRadMassFlux,binnedMass).flatten())
        hfrVel.close()

    if savePNG:
        vmax = np.max(binnedMass)
        vmin=vmax*1e-3
        plt.figure()
        plt.imshow(binnedMass,cmap='inferno',norm=LogNorm())
        plt.colorbar()
        plt.savefig(annotationFileDir_Mass+"_"+galName+"_Mass_"+str(Nsnap)+".png")
        plt.close()

    hfMass=h5py.File(annotationFileDir_Mass+"_"+galName+"_Mass_"+str(Nsnap)+".hdf5",'w')
    hfMass.create_dataset('imageName',data=image_name)
    hfMass.create_dataset('annotation',data=binnedMass.flatten())
    hfMass.close()

    binnedMass[binnedMass==0]=1e-10
    if savePNG:
        vmax = np.abs(np.max(np.divide(binnedPhiMassFlux,binnedMass)))
        vmin=0
        plt.figure()
        plt.imshow(np.divide(binnedPhiMassFlux,binnedMass),cmap='seismic')
        plt.colorbar()
        plt.savefig(annotationFileDir_RC+"_"+galName+"_RC_"+str(Nsnap)+".png")
        plt.close()

    hfMass=h5py.File(annotationFileDir_RC+"_"+galName+"_RC_"+str(Nsnap)+".hdf5",'w')
    hfMass.create_dataset('imageName',data=image_name)
    hfMass.create_dataset('annotation',data=np.divide(binnedPhiMassFlux,binnedMass).flatten())
    hfMass.close()
