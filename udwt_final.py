""" Numpy """
import numpy as np

"""Pandas"""
import pandas as pd


"""Matplotlib"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.units as munits
import matplotlib.ticker
from   cycler import cycler
import datetime

"""Seaborn"""
import seaborn as sns

""" Wavelets """
import pywt

""" Scipy """
import scipy.io
from scipy.io import savemat

"""Sort files in folder"""
import natsort

""" Load files """
from   spacepy import pycdf
import pickle
import glob
import os



def plot_pretty(dpi=175,fontsize=9):
    # import pyplot and set some parameters to make plots prettier
    plt.rc("savefig", dpi=dpi)
    plt.rc("figure", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in') 
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5) 
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5) 
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [0.5, 1.1])
    return




    """1st function: Load and split Magnetic field and particle PSP data"""

def split_data(year, res_sec, time):

    """ 
    Load and split Magnetic field and particle PSP data

    Inputs:
    year:
    res_sec: 
    time:

    Outputs:
    Saved files in folder
    """
    from scipy.io import savemat
    ### Load distance dataframe ###
    file_to_read = open(r"C:\Users\nikos.000\Current_sheets\PSP_distance.dat", "rb")

    distance = pickle.load(file_to_read)

    file_to_read.close()

    dist = distance.resample('60s').interpolate(method='linear')

    for ii in range(len(year)):

        if   ii   ==0:
            month = ['11','12']
        elif ii   ==1:
            month = ['03','04','08', '09']
        elif   ii ==2:
            month = ['01','02','05', '06','08','09','10']
        elif ii   ==3:   
            month = ['01','02','03','04','05', '06','07','08']

        for kk in range(len(month)):

            what_is   = []
            new_mean1 = []

            ### Load magnetic field dataframe ###
            file_to_read = open(r"C:\Users\nikos.000\coh_struct_distance\data\mag_SC\_merged\_"+year[ii]+"\_"+month[kk]+"_"+year[ii]+".dat", "rb")
            B            = pickle.load(file_to_read)
            file_to_read.close()

            ### Downsample magnetic field to desired cadence ###                  
            B = B.resample(res_sec).first().interpolate(method='linear')

            ### Load SPC dataframe ###
            file_to_read = open(r"C:\Users\nikos.000\coh_struct_distance\data\SPC\_merged\_"+year[ii]+"\_"+month[kk]+"_"+year[ii]+".dat", "rb")
            V            = pickle.load(file_to_read)
            file_to_read.close()



            ### Make B field dataframe to begin at the same time as SPC ###                    
            nindex  = B.index
            V       = V.reindex(V.index.union(nindex)).interpolate(method='linear').reindex(nindex)



            ### In order to select specific windows ####
            lag = (B.index[1]-B.index[0])/np.timedelta64(1,'s')
            window = int(3600*lag)

            ### Start main loop ###
            for i in range(1,int(len(B)/window)):
                Vnn  = V[(i-1)*window:(i)*window]#.values
                newV = Vnn.Vr.values.astype('str')
                if len(newV)==0:
                    print('nan')
                #elif sum(newV=='nan')/len(newV)< 2/3:  ### We want at least 2/3 of measurements to be sound samples of Vsw
                elif sum(newV=='nan')/len(newV)< 1/3:  ### We want at least 2/3 of measurements to be sound samples of Vsw
                    Bnn = B[(i-1)*window:(i)*window]#.values

                    r4       = dist.index.unique().get_loc(Bnn.index[0], method='nearest');
                    r4a      = dist.index.unique().get_loc(Bnn.index[-1], method='nearest');
                    new_dist = np.mean(dist[r4:r4a].values)

                    spc  = {"time": time,
                            "Vr"  : Vnn.Vr.values,
                            "Vt"  : Vnn.Vt.values,
                            "Vn"  : Vnn.Vn.values}
                    Bmag = np.sqrt(Bnn.Br.values**2 + Bnn.Bt.values**2 + Bnn.Bn.values**2 )
                    

                    flds = {"time": time,
                            "Bmag": Bmag,
                            "Br"  : Bnn.Br.values,
                            "Bt"  : Bnn.Bt.values,
                            "Bn"  : Bnn.Bn.values, 
                            }
                    au  = {"au": new_dist
                            }

                    savemat(r"C:\Users\nikos.000\prepare_matlab_files\anis_perp_par_dist\data\FIELDS\_"+str(year[ii])+"\_"+str(month[kk])+"\_numb_"+"_"+str(i-1)+".mat", flds)
                    savemat(r"C:\Users\nikos.000\prepare_matlab_files\anis_perp_par_dist\data\SPC\_"+str(year[ii])+"\_"+str(month[kk])+"\_numb_"+"_"+str(i-1)+".mat", spc)
                    savemat(r"C:\Users\nikos.000\prepare_matlab_files\anis_perp_par_dist\data\distance\_"+str(year[ii])+"\_"+str(month[kk])+"\_numb_"+"_"+str(i-1)+".mat", au)

                else:
                    print('not good')

    return



"""2nd function: Load and split Magnetic field and particle PSP data"""
def get_file_names(ii, kk, year, month):

    """ 
    Get names of files inside the folder

    Inputs:
    ii:    Select month out of list
    kk:    Select year out of list
    year:  A list containg the years in form of strings (e.g. ['2011','2012'])
    month: A list containg the months in form of strings (e.g. ['11','12'])

    Outputs:

    Saved files in folder

    """

     ### Sort file names
    target_path_B = r'C:\Users\nikos.000\prepare_matlab_files\anis_perp_par_dist\data\FIELDS\_'+str(year[ii])+'\_'+str(month[kk])  #path file
    file_names_B  = glob.glob(target_path_B+os.sep+'*.mat') 

    target_path_V = r'C:\Users\nikos.000\prepare_matlab_files\anis_perp_par_dist\data\SPC\_'+str(year[ii])+'\_'+str(month[kk])  #path file
    file_names_V  = glob.glob(target_path_V+os.sep+'*.mat') 

    target_path_D = r'C:\Users\nikos.000\prepare_matlab_files\anis_perp_par_dist\data\distance\_'+str(year[ii])+'\_'+str(month[kk])  #path file
    file_names_D  = glob.glob(target_path_D+os.sep+'*.mat') 

    return natsort.natsorted(file_names_B), natsort.natsorted(file_names_V), natsort.natsorted(file_names_D)
    
"""3rd function: load data """
def load_data(file_names_B, file_names_V, file_names_D, start_point, hmany):

    """ 
    Load data using output of previous function

    Inputs:

    file_names_B:    File names for magnetic field data
    file_names_V:    File names for Vsw data
    file_names_D:    File names for PSP distance data
    start_point :    Select starting file name to use
    h_many      :    How many files do you need
    
    Outputs:

    Btotal: [t, Bmag, Br, Bt, Bn]
    Vtotal : [t, Vmag, Vr, Vt, Vn, Dist]
    """
    length1  = len(file_names_B)
    for i in range(start_point,start_point + hmany):
        if i<length1:
            B = scipy.io.loadmat(file_names_B[i])
            V = scipy.io.loadmat(file_names_V[i])
            D = scipy.io.loadmat(file_names_D[i])

            if i==start_point:
                t    = B[list(B)[3]][0]; Bmag = B['Bmag'][0]; Dist   = D['au'][0];

                Br   = B['Br'][0]; Bt  = B['Bt'][0]; Bn = B['Bn'][0];

                Vr   = V['Vr'][0]; Vt = V['Vt'][0]; Vn = V['Vn'][0];
            else:
                t   = np.append(t, B[list(B)[3]][0]);  Bmag = np.append(Bmag, B['Bmag'][0]); Dist   = np.append(Dist, D['au'][0])

                Br   = np.append(Br, B['Br'][0]); Bt   = np.append(Bt, B['Bt'][0]);  Bn   = np.append(Bn, B['Bn'][0]);
                Vr   = np.append(Vr, V['Vr'][0]); Vt   = np.append(Vt, V['Vt'][0]);  Vn   = np.append(Vn, V['Vn'][0]);

    Vmag = np.sqrt(Vr**2 + Vt**2 + Vn**2)    
    Btotal = [t, Bmag, Br, Bt, Bn]
    Vtotal = [t, Vmag, Vr, Vt, Vn, Dist]
   

    return Btotal, Vtotal




""" 4th function: Perform the UDWT transform"""


def UDWT(Btotal, wname, Lps, Hps, edge_eff, norm_T_F):
    """
    Perform Undecimated discrete wavelet transform 

    Inputs:

    Btotal  : List containing, Btotal =[time, Bmag, Br, Bt, Bn]
    wname   : Wavelet name, type  str
    Lps     : Low pass filter phase shift for level 1 wavelet
    Hps     : High pass filter phase shift for level 1 wavelet 
    edge_eff: Remove edge effects or not?

    Outputs:

    Apr   : The approximations for every level and componentof the magnetic field
    Swd   : The approximations for every level and componentof the magnetic field
    pads  : Length of pads 
    nlevel: Level of decomposition

    """

    names = ['Br', 'Bt', 'Bn'] # Used Just to name files
    
    Br   =  Btotal[2]; Bt =  Btotal[3]; Bn =  Btotal[4]
    

    
    ## Set parameters needed for UDWT
    samplelength = len(Br)

    # If length of data is odd, turn into even numbered sample by getting rid 
    # of one point

    if np.mod(samplelength,2)>0:
        #Bmag = Bmag[0:-1]
        Br = Br[0:-1]
        Bt = Bt[0:-1]
        Bn = Bn[0:-1]


    # edge extension mode set to periodic extension by default with this
    # routine in the pywt.
    
    pads = 2**(np.ceil(np.log2(abs(samplelength))))-samplelength  # for edge extension, This function 
                                                                # returns 2^{ the next power of 2 }for input: samplelength

    ## Do the UDWT decompositon and reconstruction
    keep_all = {}
    for m in range(3):

        # Gets the data size up to the next power of 2 due to UDWT restrictions
        # Although periodic extension is used for the wavelet edge handling we are
        # getting the data up to the next power of 2 here by extending the data
        # sample with a constant value
    
        if (m==0):
            y = np.pad(Br,pad_width = int(pads/2) ,constant_values=np.nan)
        elif (m==1):
            y = np.pad(Bt,pad_width = int(pads/2) ,constant_values=np.nan)
        else:
             y = np.pad(Bn,pad_width = int(pads/2) ,constant_values=np.nan)
            


        # Decompose the signal using the UDWT

        nlevel = min(pywt.swt_max_level(y.shape[-1]), 10)  # Level of decomposition, impose upper limit 10
        Coeff  = pywt.swt(y, wname, nlevel, norm = norm_T_F)    # List of approximation and details coefficients 
                                                           # pairs in order similar to wavedec function:
                                                           # [(cAn, cDn), ..., (cA2, cD2), (cA1, cD1)]
        # Assign approx: swa and details: swd to 
        swa  = np.zeros((len(y),nlevel))
        swd  = np.zeros((len(y),nlevel))
       
        for o in range(nlevel):
            
            swa[:,o]  = Coeff[o][0]
            swd[:,o]  = Coeff[o][1]

       
        mzero          = np.zeros(np.shape(swd))
        A              = np.zeros(np.shape(swd))
        D              = np.zeros(np.shape(swd))
        coeffs_inverse = list(zip(swa.T,mzero.T))
        
        # Reconstruct all the approximations and details at all levels
        invers_res     = pywt.iswt(coeffs_inverse, wname, norm = norm_T_F)
        
        for pp in range(nlevel):
            swcfs           = np.zeros(np.shape(swd))
            swcfs[:,pp]     = swd[:,pp]
            coeffs_inverse2 = list(zip(np.zeros((len(swa),1)).T , swcfs.T))

            D[:,pp]         = pywt.iswt(coeffs_inverse2, wname, norm = norm_T_F)
            
        for jjj in range(nlevel-1,-1,-1):
            if (jjj==nlevel-1):
                A[:,jjj] = invers_res
            else:
                A[:,jjj] = A[:,jjj+1] + D[:,jjj+1]

            # *************************************************************************
        # VERY IMPORTANT: LINEAR PHASE SHIFT CORRECTION
        # *************************************************************************
        # Correct for linear phase shift in wavelet coefficients at each level. No
        # need to do this for the low-pass filters approximations as they will be
        # reconstructed and the shift will automatically be reversed. The formula
        # for the shift has been taken from Walden's paper, or has been made up by
        # me (can't exactly remember) -- but it is verified and correct.
        # *************************************************************************

        for j in range(1,nlevel+1):

            shiftfac = Hps*(2**(j-1));  
            
            for l in range(1,j):
                shiftfac = int(shiftfac + Lps*(2**(l-2))*((l-2)>=0)) 
                
            swd[:,j-1]   = np.roll(swd[:,j-1],shiftfac)
            
            flds = {"A": A.T,
                    "D": D.T,
                    "swd"  : swd.T
                    }

        keep_all[str(names[m])] = flds 
    
    
    
    # 1) Put all the files together into a cell structure
    Apr = {}
    Swd = {}

    pads  = int(pads)
    for kk in range(3):
        
        A              = keep_all[names[kk]]['A']
        Apr[names[kk]] = A[:,int(pads/2):len(A)-int(pads/2)]

        swd            = keep_all[names[kk]]['swd']
        Swd[names[kk]] = swd[:,int(pads/2):len(A)-int(pads/2)]

    # Returns filters list for the current wavelet in the following order
    wavelet       = pywt.Wavelet(wname)
    [h_0,h_1,_,_] = wavelet.inverse_filter_bank
    filterlength  = len(h_0)
    
    if edge_eff:
        # 2)  Getting rid of the edge effects; to keep edges skip this section

        for j in range(1,nlevel+1):

            extra = int((2**(j-2))*filterlength) # give some reasoning for this eq

            for m in range(3):
                # for approximations
                Apr[names[m]][j-1][0:extra]   = np.nan
                Apr[names[m]][j-1][-extra:-1] = np.nan

                # for details
                Swd[names[m]][j-1][0:extra]   = np.nan
                Swd[names[m]][j-1][-extra:-1] = np.nan
    
    return  Apr, Swd, pads, nlevel

""" 5th function: Calculates the mean magnetic field directional unit vector"""

def anisotropy1(Br,Bt, Bn, Vunit):

    """ 
    # Calculates the mean magnetic field directional unit vector 'parDir' the 
    # two perpendicular unit vectors 'perpA' and 'perpB' to this using the
    # solar wind velocity unit vector 'vel' and finally the angle 'BVangle' of
    # 'parDir' to 'vel'.
    # 
    # Inputs:-
    # 
    # Br,Bt,Bn : Components of the magnetic field
    # Vunit    : Components of the Solar wind velocity 
    # 
    # Outputs:-
    # 
    # parDir: unit vector parallel to magnetic field direction
    # perpA: unit vector = crossproduct of 'parDir' and 'vel'
    # perpB: unit vector = crossproduct of 'parDir' and 'perpA'
    # BVangle: angle in degrees between 'parDir' and 'vel'

    """
    
    Vmag  = np.array(Vunit[1]);  Vr = np.array(Vunit[2])/Vmag;  Vt = np.array(Vunit[3])/Vmag; Vn = np.array(Vunit[4])/Vmag;
    
    
   # diff = len(Br) - len(Vr)
    
    Vr = np.resize(Vr,len(Br)); Vt = np.resize(Vt,len(Br)); Vn = np.resize(Vn, len(Br))

    Bmag  = np.sqrt(Br*Br + Bt*Bt+ Bn*Bn)

    parDir1 = Br/Bmag
    parDir2 = Bt/Bmag
    parDir3 = Bn/Bmag

    parDir  = [parDir1, parDir2, parDir3]


    # angle between mean solar wind velcity unit vector 
    #and  magnetic field unit vector

    cosBVangle = parDir1*Vr + parDir2*Vt + parDir3*Vn
    BVangle    = np.arccos(cosBVangle)
    BVangle    = BVangle*(360/(2*np.pi)) # conversion from radians to degrees


    # create new orthonormal basis from parDir, parDir X Vel
    # and parDir X (parDir X Vel). Here are the additional perp vectors. 

    perpA1 = parDir2*Vn - parDir3*Vt
    perpA2 = parDir3*Vr - parDir2*Vn
    perpA3 = parDir1*Vt - parDir1*Vr

    # We need to normalise by magnitude here as the resultant perpA vector will
    # not be a unit vector unless we do so. This is needed because 'vel' and
    # 'pardir', even though they are unit vectors, are not necessarily
    # orthogonal to each other and hence a cross product will produce something
    # which is not exactly norm=1 (which is expected for a unit vector).

    perpAmag = np.sqrt(perpA1*perpA1 + perpA2*perpA2 + perpA3*perpA3)

    perpA1 = perpA1/perpAmag
    perpA2 = perpA2/perpAmag
    perpA3 = perpA3/perpAmag

    perpA = [perpA1,perpA2,perpA3]



    # no need to normalise by magnitude here as the two vectors being crossed
    # are exactly orthogonal hence this will produce an exact unit vector.

    perpB1 = parDir2*perpA3 - parDir3*perpA2
    perpB2 = parDir3*perpA1 - parDir1*perpA3
    perpB3 = parDir1*perpA2 - parDir2*perpA1

    perpB=[perpB1, perpB2, perpB3]

    return parDir,perpA, perpB, BVangle


""" 6th function:
 Project fluctuations given by wavelet details, parallel and perp to the
 scale dependent background field direction given by 'parDir'
"""

def anisotropyProject(parDir, perpA, perpB, details1,details2, details3):
    """
     ------------
     Projection function which takes the details and the unit vector from the 
     approximations and projects the details onto the unit vector. This will 
     churn out the parallel and perpendicular details.

     Inputs:-

     parDir:
     perpA:
     perpB:
     details:

     Outputs:-

     Bpar:
     BperpA:
     BperpB:
     BperpMag:

    **************************************************************************
    """

    Bpar     = details1*parDir[0] + details2*parDir[1] + details3*parDir[2]
    BperpA   = details1*perpA[0]  + details2*perpA[1]  + details3*perpA[2]
    BperpB   = details1*perpB[0]  + details2*perpB[1]  + details3*perpB[2]
    BperpMag = np.sqrt(BperpA*BperpA + BperpB*BperpB)
    BtotMag  = np.sqrt(Bpar*Bpar+ BperpA*BperpA + BperpB*BperpB)

    return BtotMag, Bpar, BperpA, BperpB, BperpMag


def project_2_Vsw(Apr, Swd, nlevel, pads, wname, Vunit):
    # Decompose detail coefficients at all levels into perp and par components
    names =['Br', 'Bt', 'Bn']
    
    coefsBVangle = {}
    coefspar     = {}
    coefsperpA   = {}
    coefsperpB   = {}
    coefsperpMag = {}
    coefsBtotMag = {}
    for i in range(nlevel):

        # Using an external function, construct orthonormal basis comprising of 
        # background field direction, and two perpendicular vectors using the solar
        # wind plasma flow velocity direction.
        # This function will also output the angle between the magnetic field and
        # the background solar wind plasma flow velocity direction which we can
        # assume to be constant.


        parDir,perpA,perpB,BVangle = anisotropy1(Apr[names[0]][i], Apr[names[1]][i],
            Apr[names[2]][i],Vunit)

        # Project fluctuations given by wavelet details, parallel and perp to the
        # scale dependent background field direction given by 'parDir'

        BtotMag, Bpar, BperpA, BperpB,BperpMag = anisotropyProject(parDir,perpA,perpB,
                                       Swd[names[0]][i], 
                                       Swd[names[1]][i],
                                       Swd[names[2]][i])


        coefsBVangle[str(i)] = BVangle
        coefspar[str(i)]     = Bpar
        coefsperpA[str(i)]   = BperpA
        coefsperpB[str(i)]   = BperpB
        coefsperpMag[str(i)] = BperpMag
        coefsBtotMag[str(i)] = BtotMag

        
    return coefsBtotMag, coefspar, coefsperpA,coefsperpB, coefsperpMag, coefsBVangle



def function_4(coefsBtotMag, coefspar, coefsperpA, coefsperpB, coefsperpMag, coefsBVangle, anglebins, nlevel):
# Calculating length of vector at each wavelet stage, excluding NaNs

    lengthofsamples = np.zeros(nlevel)
    for j in range(nlevel):
        a = np.isnan(coefspar[str(j)])
        b = np.where(a==0)
        lengthofsamples[j]= len(b[0])

    ## Getting the indices of the arrays at different angles

    absize       = 2#len(anglebins)
    #print(absize)
    angleindices = {}

    for m in range(absize):   
        if m ==0:
            startangle = anglebins[0]#-15.0
            endangle   = anglebins[1]#+15.0
        else:
            startangle = anglebins[2]#-15.0
            endangle   = anglebins[3]#+15.0

        angleindices1 = {}
        for k in range(nlevel):
            if m==0:
                conditional1 = (coefsBVangle[str(k)]>startangle) & (coefsBVangle[str(k)]< endangle)
                conditional2 = (coefsBVangle[str(k)]>160) & (coefsBVangle[str(k)]< 180)

                angleindices1[str(k)]= np.where( conditional1 | conditional2 )[0].astype(int)
            else:
                angleindices1[str(k)]= np.where((coefsBVangle[str(k)]>startangle) & (coefsBVangle[str(k)]< endangle))[0].astype(int)

        angleindices[str(m)] = angleindices1


    return angleindices, absize, lengthofsamples 

## Calculating power spectral density with errors + magnetic compresib..

def get_quants_2_plot(nlevel,qorder, alternative_comp, wname, dt, absize, angleindices,coefsBtotMag, coefspar,coefsperpA, coefsperpB,coefsperpMag):

    p_Bmag_order = {}
    for j in range(len(qorder)):
        p_BtotMag  = np.zeros((nlevel,absize))
        dp_BtotMag = np.zeros((nlevel,absize))
        for m in range(absize):
              for k in range(nlevel):

                cw          = (coefsBtotMag[str(k)][angleindices[str(m)][str(k)]])**qorder[j]
                p_BtotMag[k,m]  = (np.nanmean(cw))*(dt*2)

        p_Bmag_order['qorder_'+str(qorder[j])] = p_BtotMag


    
    ## Calculating power spectral density for par with errors    

    p_par_order = {}
    for j in range(len(qorder)):
        p_par  = np.zeros((nlevel,absize))
        dp_par = np.zeros((nlevel,absize))
        for m in range(absize):
            for k in range(nlevel):
                cw = abs(coefspar[str(k)][angleindices[str(m)][str(k)]])**qorder[j]
                p_par[k,m]   = (np.nanmean(cw))*(dt*2)

        p_par_order['qorder_'+str(qorder[j])] = p_par


    ## Calculating power spectral density for perpA with errors

    p_perpA  = np.zeros((nlevel,absize))
    dp_perpA = np.zeros((nlevel,absize))

    for m in range(absize):
        for k in range(nlevel):
            cw = coefsperpA[str(k)][angleindices[str(m)][str(k)]]*coefsperpA[str(k)][angleindices[str(m)][str(k)]];
            p_perpA[k,m]   = (np.nanmean(cw))*(dt*2)


    ## Calculating power spectral density for perpB with errors

    p_perpB  = np.zeros((nlevel,absize))
    dp_perpB = np.zeros((nlevel,absize))

    for m in range(absize):
        for k in range(nlevel):
            cw = coefsperpB[str(k)][angleindices[str(m)][str(k)]]*coefsperpB[str(k)][angleindices[str(m)][str(k)]]
            p_perpB[k,m]   = (np.nanmean(cw))*(dt*2)



    ## Calculating power spectral density for perpMag with errors


    p_perpMag_order = {}
    for j in range(len(qorder)):
        p_perpMag  = np.zeros((nlevel,absize))
        dp_perpMag = np.zeros((nlevel,absize))
        for m in range(absize):
            for k in range(nlevel):
                cw = abs(coefsperpMag[str(k)][angleindices[str(m)][str(k)]])**qorder[j]
                p_perpMag[k,m]   = (np.nanmean(cw))*(dt*2)

        p_perpMag_order['qorder_'+str(qorder[j])] = p_perpMag

    # Calculate magnetic compressibility
    if alternative_comp:
        magcom = np.zeros((nlevel,absize))
        dmagcom = np.zeros((nlevel,absize))

        for m in range(absize):
            for k in range(nlevel):

                cwperp = coefsperpMag[str(k)][angleindices[str(m)][str(k)]]*coefsperpMag[str(k)][angleindices[str(m)][str(k)]]
                cwpar = coefspar[str(k)][angleindices[str(m)][str(k)]]*coefspar[str(k)][angleindices[str(m)][str(k)]]
                cw = cwpar/(cwpar+cwperp)
                magcom[k,m] = (np.nanmean(cw))
    

    else:

        magcom = np.zeros((nlevel,absize));
        dmagcom = np.zeros((nlevel,absize));

        for m in range(absize):
            for k in range(nlevel):

                magcom[k,m] = (p_par[k,m]/(p_perpMag[k,m]+p_par[k,m]))
                dmagcom[k,m] = ((dp_par[k,m])*(dp_par[k,m]))*((p_par[k,m])*(p_par[k,m]))
           
    scale     = 2**np.arange(1,nlevel);
    frequency = pywt.scale2frequency(wname,scale,dt)
    
    return p_Bmag_order, scale, frequency ,p_BtotMag, dp_BtotMag, p_par_order, dp_par, p_perpA,dp_perpA, p_perpB, dp_perpB, p_perpMag_order, dp_perpMag, magcom, dmagcom