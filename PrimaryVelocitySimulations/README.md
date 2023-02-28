I wrote this program to test the efficiency of different observational strategies for detecting hot Jupiters. I simulated 5 epoch Keck/NIRSPEC data sets with two variable parameters, (1) the relative Doppler shift between the stellar and telluric spectrum and (2) the relative Doppler shift between the planetary and stellar spectrum. The Gaussian models fit and plotted in this program were used to identify which simulated data sets showed strong and clear planetary signals.
[https://iopscience.iop.org/article/10.3847/1538-3881/abf7b9/meta]



# Files 
SimulateSpec.py - Program I wrote to simulate infrared spectra from Keck at a number of different observations. Stellar and planetary spectral models are combined at the right contrast and right Doppler shifts, broadened and interpolated onto the same wavelength axis as real observational data.

goodnessparam.py - This code analyzes hundreds of simulated results, fittings Gaussians to the suspected planetary detection peaks, and reporting whether the Gaussians accurately detected the simulated planet or did not. This was done to predict the efficiency of different observational strategies. 


# Figures
[gaussianfitparams_noninv.pdf](https://github.com/cbuzard/ScienceProjects/files/10833907/gaussianfitparams_noninv.pdf)
