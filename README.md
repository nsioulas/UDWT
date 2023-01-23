# UDWT
This file decomposes the vector magnetic field fluctuations onto the vector background magnetic field using the undecimated discrete wavelet decomposition. The scale-by-scale 'details' and 'approximations' provide the scale-by-scale fluctuations and background field, respectively. This type of anisotropy study is historically called 'variance anisotropy'. This is basically the anisotropy study of magnetic field quantities rather than the wavevector 'k_i' values that it depends upon.



# Where to find data?

  The datasaet used for this project can be found here:

  https://drive.google.com/drive/folders/1Wh7UDQU9YldhfzyVOGlrUiZJjTW8aTL0?usp=sharing

  This dataset includes magnetic field and particle data ata various heliospheric distances, separated into 1-Hour intervals. The format of the files is .mat. 
  This is because those files were meant to be used in Matlab. However, one can load those files in pythin with Scipy and it doesn't seem to be significantly slower than other more frequently used methods.



