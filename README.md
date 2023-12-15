# CoNNGaFit
<b>Co</b>nvoltuional <b>N</b>eural <b>N</b>etwork for <b>Ga</b>laxy <b>Fit</b>ting. Train and use convolutional neural networks to fit various galactic parameters to synthetic HI images. Training images and annotations for the FIRE simulations can be generated using VeryObservableFIRE
<b>V<\b>ery<b>O<\b>bservable<b>F<\b>IRE is a software suite for creating synthetic spectra from the FIRE simulations. To use, modify param_template.py with the relevant information regarding your input data and desired outputs. Currently only has support for HI 21cm, H alpha, and the Sodium I doublet. Additionally, if RunBinfire is enabled in the paramter file, annotation files containing information regarding the dynamics and structure of the simulated galaxies will be created for use training Neural Networks, specifically with CoNNGaFit.