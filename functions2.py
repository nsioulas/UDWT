#!/usr/bin/env python
# coding: utf-8

# #  Necessary Packages

# In[1]:


import os
from spacepy import pycdf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math
import matplotlib.dates as mdates
import pandas as pd 
import matplotlib.units as munits
from cycler import cycler
import pickle


# In[15]:


#   1) Nan removal

# Make process faster -Define functions

### define functions #########

import numba
from numba import jit,njit,prange,objmode

@jit(nopython=True)
def nan_removal(b,replace1,window,desired_min):
    minimum = min(b)
    if replace1:
        for i in range(len(b)):
            if (b[i]== minimum) or (b[i]<desired_min):
                b[i] = np.nanmedian(b[i-window:i+window]) #np.nan
    else:
        for i in range(len(b)):
            if (b[i]== minimum) or (b[i]<desired_min):
                b[i] = np.nan#np.nanmedian(b[i-window:i+window]) #np.nan
    return b
################################################################################################3


#   2) Hampel Algorithm

from numba import jit,njit,prange

@jit(nopython=True, parallel=True)
def hampel_filter_forloop_numba(input_series, window_size, n_sigmas=3):
    
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826 # scale factor for Gaussian distribution
    indices = []
    
    for i in range((window_size),(n - window_size)):
        x0 = np.nanmedian(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)
    
    return new_series, indices


##########################################################################################################


#  3) BEst fit in loglog

### define fitting functions ###

import scipy as sc
from scipy.optimize import curve_fit


##### define the Function for best linear fit #####
# solve for a and b
def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b


def powlaw(x, a, b) : 
    return a * np.power(x, b) 
def linlaw(x, a, b) : 
    return a + x * b

def curve_fit_log(xdata, ydata) : 
    """Fit data to a power law with weights according to a log scale"""
    # Weights according to a log scale
    # Apply fscalex
    xdata_log = np.log10(xdata)
    # Apply fscaley
    ydata_log = np.log10(ydata)
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
    #print(popt_log, pcov_log)
    # Apply fscaley^-1 to fitted data
    ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
    # There is no need to apply fscalex^-1 as original data is already available
    return (popt_log, pcov_log, ydatafit_log)
#################################################################################################################

#   4)Interoplate
def interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out



# 5) Last resort for temperature:
@jit(nopython=True)
def weird_temp(b,window,desired_min):
    minimum = min(b)
    for i in range(len(b)):
        if  (b[i]<desired_min):
            b[i] = np.nanmedian(b[i-window:i+window]) #np.nan
    return b

# 6) Choose events for cs:


@jit(nopython=True)
def current_sheets(der_B,der_ratio,der_beta,thresh_B_psp,thresh_beta_psp,thresh_ratio_psp):
    indexx = np.zeros(len(der_ratio))
    for i in range(len(der_ratio)):
        if der_B[i]<= thresh_B_psp:
            if (der_beta[i]>= thresh_beta_psp) or (der_ratio[i]<= thresh_ratio_psp):
                indexx[i] = 1
    return indexx


#7)  Scatter plot with median

@jit(nopython=True)
def scatter_plot_median(distances,rates,binsa,sum1,average,bin_centersout):
    for z in range(1,len(bin_centersout)):
        for i in range(len(rates)):
            if z==1:
                if distances[i] <= binsa[z-1]:
                    sum1[i] = rates[i]
            else:
                if (distances[i] >= binsa[z-1]) & (distances[i] <= binsa[z]): 
                    sum1[i] = rates[i]
        if np.sum(sum1[sum1>0]):      
            average[z] = np.median(sum1[sum1>0])
        else:
             average[z] = np.nan
    return average

@jit(nopython=True)
def scatter_plot_mean(distances,rates,binsa,sum1,average,bin_centersout):
    for z in range(1,len(bin_centersout)):
        for i in range(len(rates)):
            if z==1:
                if distances[i] <= binsa[z-1]:
                    sum1[i] = rates[i]
            else:
                if (distances[i] >= binsa[z-1]) & (distances[i] <= binsa[z]): 
                    sum1[i] = rates[i]
        if np.sum(sum1[sum1>0]):      
            average[z] = np.mean(sum1[sum1>0])
        else:
             average[z] = np.nan
    return average
	
	
	
# 8) Gang li method for current sheet identification
	
@jit(nopython=True)
def angles_li(ar,at,an,z):
    ang = np.zeros(len(ar))
    mag = np.sqrt(ar**2 +at**2 +an**2)
    for i in range(z,len(ar)):
        ang[i] = np.arccos(((ar[i]*ar[i-z]) + (at[i]*at[i-z]) + (an[i]*an[i-z])) /( mag[i] * mag[i-z]))*(180/np.pi)
    return ang, mag
	
# 9) PDF plot

def histogram(quant, bins2,logx):
    nout = []
    bout = []
    errout=[]
    if logx == True:
        binsa = np.logspace(np.log10(min(quant)),np.log10(max(quant)),bins2)
    else:
        binsa = np.linspace((min(quant)),(max(quant)),bins2)
            
    histoout,binsout = np.histogram(quant,binsa,density=True)
    erroutt = histoout/np.float64(np.size(quant))
    erroutt = np.sqrt(erroutt*(1.0-erroutt)/np.float64(np.size(quant)))
    erroutt[0:np.size(erroutt)] = erroutt[0:np.size(erroutt)] /(binsout[1:np.size(binsout)]-binsout[0:np.size(binsout)-1])

    bin_centersout   = binsout[:-1] + np.log10(0.5) * (binsout[1:] - binsout[:-1])
        
    for k in range(len(bin_centersout)):
        if (histoout[k]!=0.):
            nout.append(histoout[k])
            bout.append(bin_centersout[k])
            errout.append(erroutt[k])
    return nout, bout,errout
	
	
# 10) Alfvenicity functions
	
@njit(parallel=True)
def func_DB_denom(ar,at,an,tau):
    DBtotal= np.zeros((len(ar)))
    for i in prange(tau,len(ar),1):
        DBtotal[i] = np.sqrt((ar[i]- ar[i -tau])**2 + (at[i]- at[i - tau])**2 + (an[i]- an[i-tau])**2)
    return DBtotal
@njit(parallel=True)
def func_DB_numer(B,tau):
    DBtotal= np.zeros((len(B)))
    for i in prange(tau,len(B),1):
        DBtotal[i] = abs(B[i]-B[i-tau])
    return DBtotal
	
# 11) PDF

def pdf(val, bins,loglog,density):
    nout  =[]
    bout  =[]
    errout=[]

    if loglog ==1:
        binsa = np.logspace(np.log10(min(val)),np.log10(max(val)),bins)
    else:
        binsa = np.linspace((min(val)),(max(val)),bins)
        
    if density ==1:
        numout, binsout, patchesout = plt.hist(val,density= True,bins=binsa, alpha = 0)
        
    else:
        numout, binsout, patchesout = plt.hist(val,density= False,bins=binsa, alpha = 0)
        
    if loglog ==1:
        bin_centers = binsout[:-1] + np.log10(0.5) * (binsout[1:] - binsout[:-1])
    else:
        bin_centers = binsout[:-1] +         (0.5) * (binsout[1:] - binsout[:-1])
        
    if density ==1:
        histoout,edgeout=np.histogram(val,binsa,density= True)
    else:
        histoout,edgeout=np.histogram(val,binsa,density= False)
    
    erroutt = histoout/np.float64(np.size(val))
    erroutt = np.sqrt(erroutt*(1.0-erroutt)/np.float64(np.size(val)))
    erroutt[0:np.size(erroutt)] = erroutt[0:np.size(erroutt)] /(edgeout[1:np.size(edgeout)]-edgeout[0:np.size(edgeout)-1])
 
    for i in range(len(numout)):
        if (numout[i]!=0.):
            nout.append(numout[i])
            bout.append(bin_centers[i]) 
            errout.append(erroutt[i])
    
    return  np.array(bout), np.array(nout), np.array(errout)


	
# 12) PVI

#################-- PVI AS A FUNCTION OF TIME for magnetic field --#####################
import numba
from numba import jit,njit,prange

@numba.jit(nopython=True, parallel=True)
def func_DB(ar, at, an, tau):
    
    DBtotal= np.zeros((len(ar)))
    for i in prange(0,len(ar)-tau,1):
        DBtotal[i + tau] = np.sqrt((ar[i + tau]- ar[i])**2 + (at[i + tau]- at[i])**2 + (an[i + tau]- an[i])**2)
    return DBtotal


@numba.jit(nopython=True, parallel=True)
def func_PVI(DBtotal,tau,lag,hours):
    
    window = int(hours*3600/lag)                              # averaging window
    PVI    = np.zeros((len(DBtotal)))                         # create 1D array
    
    for i in prange(int(window/2),len(DBtotal)-int(window/2)): 
        PVI[i] = DBtotal[i]/np.sqrt((np.mean(DBtotal[i-int(window/2):i+int(window/2)]**2)))
    return PVI 
	
	
# 13) Find duration of interm structures




def find_duration(df,what,theta1,theta2):
    small = (df.index[1] - df.index[0])/np.timedelta64(1, 's')
    a = pd.DataFrame(df[what].values, columns = [what])
    a['condition'] = a[what].between(theta1,theta2) #(a.PVI  >= 2) #and (a.PVI  <= 6)#(a.PVI > 4)
    a['crossing']  = (a.condition != a.condition.shift()).cumsum()
    a['count']     = a.groupby(['condition', 'crossing']).cumcount(ascending=False) + 1
    a.loc[a.condition == False, 'count'] = 0
    krata        = np.zeros(len(a['count'].values))
    krata1       = np.zeros(len(a['count'].values))
    duration     = np.zeros(len(a['count'].values))
    indexs       = a['count'].values
    values       = a['condition'].values
    PVIs         = a[what].values

    for i in range(1,len(values)-1):
        if values[i]==True:
            if values[i-1]==False:
                krata[i]     = i
                krata1[i]    = i+int(indexs[i])
                time         = df.index[i+int(indexs[i])] - df.index[i]
                duration[i]  = time/ np.timedelta64(1, 's')  -small
   
    return krata, krata1, duration
	
	
# 14) Find waiting times between interm struct.

def find_wt(df,what,theta1,theta2):

    df1 = df[(df[what]>theta1) & (df[what]<theta2) ]#between(theta1,theta2) 
    a   = df1.index.values
    time =[]
    small = (df.index[1] - df.index[0])/np.timedelta64(1, 's')
    for i in range(1,len(a)):
        if (a[i]-a[i-1])/np.timedelta64(1, 's') > small:
            time.append((a[i]-a[i-1])/np.timedelta64(1, 's') - small)
    return np.array(time)
	
	
# 15) Find corel between PVI and Temp, create dataframes


def create_pvi_temp(kk, temp_min,temp_max, choose_tau):
    
    ### choose orbit ###
    string11 = ['1st','2nd','3rd','4th','5th','6th','7th','8th']
    string21 = ['1st_0.8s','2nd_0.8s','3rd_0.8s','4th_0.8s','5th_0.8s','6th_0.8s']
    ### choose tau   ###
    tau     = [1,10,100]  
    
    ### load concat magnetic field data at 0.8736 resolution    ###
    
    file_to_read = open(r"C:\Users\nikos.000\PVI\data\FIELDS_concat\_"+string11[kk]+".dat", "rb")

    dfB   = pickle.load(file_to_read)

    file_to_read.close()
    
    dfB   = dfB.resample('0.873812S').mean().interpolate(method='linear')
   
    
    ### load clean temp data                                    ###
    
    file_to_read = open(r"C:\Users\nikos.000\PVI\data\SPC_concat\clean\_"+string11[kk]+".dat", "rb")
    SPC = pickle.load(file_to_read)
    file_to_read.close()
    SPC = SPC[~SPC.index.duplicated()]
    
    ### load PVI data                                           ###
    
    file_to_read = open(r"C:\Users\nikos.000\PVI\data\PVI\_t_"+str(tau[choose_tau])+"\_"+string21[kk]+".dat", "rb")
    PVI = pickle.load(file_to_read)
    file_to_read.close()
    PVI = PVI[~PVI.index.duplicated()]

    
    ### Resample temperature acording to the index of mag field ###
    
    nindex = PVI.index
    
    T2     = SPC.reindex(SPC.index.union(nindex)).interpolate(method='linear').reindex(nindex)
    
    ### find angles of magnetic field less than 30 degrees      ###
    
    ar     = dfB.Br.values
    at     = dfB.Bt.values
    an     = dfB.Bn.values
    
    
    angle_theta = np.arccos((ar**2)/(np.sqrt(ar**2 + at**2 + an**2)*abs(ar)))*(180/math.pi)

    
    ### create a df including PVI, angle, Tmom and Tfit, id_theta ###
    df = pd.DataFrame({'DateTime': PVI.index,
                             'PVI': PVI['PVI'].values,
                            'Tmom': T2['$Tmom$'],
                            'Tfit': T2['$Tfit$'],
                            'angle': angle_theta})

    file_to_store = open(r"C:\Users\nikos.000\PVI\data\PVI_temp_corel\\_"+string11[kk]+".dat", "wb") # sto trito bale 2 opws einai twra
    pickle.dump(df, file_to_store)
    file_to_store.close()

    
    return df


# 16) sort pvi temperature
def sort_pvi_temp(temp_min, temp_max,moments,min_angle,string11):
    pvi_dir  = {} # Create pvi directory
    tem_dir  = {} # Create tem directory
    
    for kk in range(len(string11)):
        file_to_read = open(r"C:\Users\nikos.000\PVI\data\PVI_temp_corel\\_"+string11[kk]+".dat", "rb")
        df = pickle.load(file_to_read)
        file_to_read.close()
    
        ### Mask instances where angle > angle min degrees and T<Tselected   ###
        if moments:
            df_new = df.mask((df['Tmom']<temp_min) | (df['Tmom']>temp_max) |  (df['angle']>min_angle))
        else:
            df_new = df.mask((df['Tfit']<temp_min) | (df['Tfit']>temp_max) | (df['angle']>min_angle))
    
        ### Sort Temperature according to corresponding PVI values    ###
        if moments:
            temperature  = df_new.Tmom.values
        else:
            temperature  = df_new.Tfit.values
        PVI          = df_new.PVI.values
        inds         = PVI.argsort()
        sorted_Tem   = temperature[inds]
        sorted_PVI   = PVI[inds]
        
        df = pd.DataFrame({'PVI':sorted_PVI,
                           'Tem': sorted_Tem})
        
        ### Save files ### 

        if moments:
            file_to_store = open(r"C:\Users\nikos.000\PVI\data\PVI_temp_corel\final\_mom"+string11[kk]+".dat", "wb") # sto trito bale 2 opws einai twra
            pickle.dump(df, file_to_store)
            file_to_store.close()
        else:
            file_to_store = open(r"C:\Users\nikos.000\PVI\data\PVI_temp_corel\final\_fit"+string11[kk]+".dat", "wb") # sto trito bale 2 opws einai twra
            pickle.dump(df, file_to_store)
            file_to_store.close()


# 17 Concat_ resample Magnetic field data.



def concat_resample_PVI_1min(string1, perih):
    import glob
    for k in range(len(string1)):
        target_path = r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\FIELDS_RAW\_"+string1[perih]+""  # path file
        file_names  = glob.glob(target_path+os.sep+'*.dat')                                               # file names
        for i in range(len(file_names)):
            if i==0:
                file_to_read = open(file_names[i], "rb")
                B = pickle.load(file_to_read)
                file_to_read.close()
            else:
                file_to_read = open(file_names[i], "rb")
                B1 = pickle.load(file_to_read)
                file_to_read.close()
            if i>0:
                all_files =[B,B1]
                B = pd.concat(all_files)  
                
        #B_new = B.set_index('DateTime')
        file_to_store = open(r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\FIELDS_concat\_year"+string1[perih]+".dat", "wb")
        pickle.dump(B, file_to_store)
        file_to_store.close()
    

# 18) Estimate PVI and save PVI
def estimate_pvi(tau, hours, res_sec, res_sec2, string1, string11, perih, angle_max):
    import functions1 as fun
    for k in range(len(string11)):
        file_to_read = open(r"C:\Users\nikos.000\PVI_TEMPERATURE\data\FIELDS_concat\_lag_"+str(res_sec2)+"_"+string1[perih]+"_orbit_"+string11[k]+".dat", "rb")
        B = pickle.load(file_to_read)
        file_to_read.close()
    
       ### Numba does not recong DF's, convert to np.array ###

        ar     = B.Br.values
        at     = B.Bt.values
        an     = B.Bn.values

        ###  Estimate cadence of DF ###
        lag     = (B.index[1]-B.index[0])/np.timedelta64(1,'s') 
    
        DBtotal = fun.func_DB(ar, at, an, tau)
        PVI     = fun.func_PVI(DBtotal, tau, lag, hours)
        PVI_df  = pd.DataFrame({'DateTime': B.index,
                                     'PVI': PVI,
                                   'angle': B['angle'].values})
    
        PVI_df  = PVI_df.set_index('DateTime')
        
        file_to_store = open(r"C:\Users\nikos.000\PVI_TEMPERATURE\data\PVI_full_resol\_"+string1[perih]+".dat", "wb") # sto trito bale 2 opws einai twra
        pickle.dump(PVI_df, file_to_store)
        file_to_store.close()
        
        ### remove 
        PVI_df = PVI_df.mask((PVI_df['angle']>angle_max))
        ### Resaple to desired cadence ###
        PVI_df = PVI_df.resample(res_sec).max().interpolate(method='linear')
    
        file_to_store = open(r"C:\Users\nikos.000\PVI_TEMPERATURE\data\PVI_resampled\_"+string1[perih]+".dat", "wb") # sto trito bale 2 opws einai twra
        pickle.dump(PVI_df, file_to_store)
        file_to_store.close()
		
		
# 18) Concat resampled dataframes of PVI
def concat_resampled_PVI(string1,test):
    import glob
    for k in range(len(string1)):
        if test:
            target_path = r"C:\Users\nikos.000\PVI_TEMPERATURE\data\test\PVI_resampled\_"+string1[k]+""   # file path 
        else:
            target_path = r"C:\Users\nikos.000\PVI_TEMPERATURE\data\PVI_resampled\_"+string1[k]+""   # file path 
            
        file_names  = glob.glob(target_path+os.sep+'*.dat')                                      # file names
        vals  = []
        index = []
        for i in range(len(file_names)):
        
            file_to_read = open(file_names[i], "rb")
            B            = pickle.load(file_to_read)                                             # Load files
            file_to_read.close()
        
            lag     = (B.index[1]-B.index[0])/np.timedelta64(1,'s')
            keep    = int((4*3600)/lag)
            if i ==0:
                B_val = B.PVI[keep:-keep].values
                B_ind = B[keep:-keep].index.values
            else:
                B_val = B.PVI[keep:-keep].values
                B_ind = B[keep:-keep].index

            for oo in range(len(B_val)):
                vals.append(B_val[oo])
                index.append(B_ind[oo])
        
        B_new = pd.DataFrame({'DateTime':np.array(index),
                             'PVI': np.array(vals)})
        B_new = B_new[~B_new.index.duplicated(keep='first')]
        B_new = B_new.set_index('DateTime')		
                                    # concat df's and drop duplicates
        if test:
            file_to_store = open(r"C:\Users\nikos.000\PVI_TEMPERATURE\data\test\PVI_resampled\_orbit_"+string1[k]+"_merged.dat", "wb") 
            pickle.dump(B_new, file_to_store)
            file_to_store.close()
        else:
            file_to_store = open(r"C:\Users\nikos.000\PVI_TEMPERATURE\data\PVI_resampled\_orbit_"+string1[k]+"_merged.dat", "wb") 
            pickle.dump(B_new, file_to_store)
            file_to_store.close()
            
# 19) Clean energies from pyspedas
@jit(nopython=True, parallel=True)
def clean_energies_pyspedas(energies):
    for i in range(len(energies)):
        for k in range(len(energies[i])):
            if energies[i][k]<-1e1:
                energies[i][k] = np.nan
    return energies


# FIX perihelion 4th ISOIS

def concat_resample_ISOIS_ions(string1, perih):
    import glob
    for k in range(len(string1)):
        target_path = r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\ISOIS\_"+string1[perih]+"" #path file
        file_names  = glob.glob(target_path+os.sep+'*.cdf')                      # file names
        for i in range(len(file_names)):
            data = pycdf.CDF(file_names[i])
            if i==0: 
                test = data['Epoch_ChanT'][:]
                test2 = data['H_CountRate_ChanT_SP'][:]
                sum_counts =[]
                for oo in range(len(test2)):
                    sum1 = test2[oo][test2[oo]>0]
                    sum_counts.append(np.sum(sum1))
                B = pd.DataFrame({'DateTime': test,
                   '$Count-rate \ (s^{-1})$': sum_counts})

            else:
                test = data['Epoch_ChanT'][:]
                test2 = data['H_CountRate_ChanT_SP'][:]
                sum_counts =[]
                for oo in range(len(test2)):
                    sum1 = test2[oo][test2[oo]>0]
                    sum_counts.append(np.sum(sum1))
                B1 = pd.DataFrame({'DateTime': test,
                   '$Count-rate \ (s^{-1})$': sum_counts})  
            if i>0:
                all_dfs = [B,B1]
                B =pd.concat(all_dfs)   
                 
        file_to_store = open(r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\ISOIS\_"+string1[perih]+"orbit.dat", "wb") # sto trito bale 2 opws einai twra
        pickle.dump(B, file_to_store)
        file_to_store.close()

# FIX perihelion 4th ISOIS electrons

def concat_resample_ISOIS_electrons(string1, perih):
    import glob
    for k in range(len(string1)):
        target_path = r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\ISOIS\_"+string1[perih]+"" #path file
        file_names  = glob.glob(target_path+os.sep+'*.cdf')                      # file names
        for i in range(len(file_names)):
            data = pycdf.CDF(file_names[i])
            if i==0: 
                test = data['Epoch_ChanE'][:]
                test2 = data['Electron_CountRate_ChanE'][:]
                sum_counts =[]
                for oo in range(len(test2)):
                    sum1 = test2[oo][test2[oo]>0]
                    sum_counts.append(np.sum(sum1))
                B = pd.DataFrame({'DateTime': test,
                   '$Count-rate \ (s^{-1})$': sum_counts})

            else:
                test = data['Epoch_ChanE'][:]
                test2 = data['Electron_CountRate_ChanE'][:]
                sum_counts =[]
                for oo in range(len(test2)):
                    sum1 = test2[oo][test2[oo]>0]
                    sum_counts.append(np.sum(sum1))
                B1 = pd.DataFrame({'DateTime': test,
                   '$Count-rate \ (s^{-1})$': sum_counts})  
            if i>0:
                all_dfs = [B,B1]
                B =pd.concat(all_dfs)   
                 
        file_to_store = open(r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\ISOIS\_electrons"+string1[perih]+"orbit.dat", "wb") # sto trito bale 2 opws einai twra
        pickle.dump(B, file_to_store)
        file_to_store.close()

# 21)Plot Count rates ions


def plot_count_rates_ions(string1):
    file_to_read = open(r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\PVI\_Start_End_dates.dat", "rb")
    df = pickle.load(file_to_read)
    file_to_read.close()
    import matplotlib.dates as md
    string21           = ['$1^{st}$','$2^{nd}$','$3^{rd}$','$4^{th}$','$5^{th}$','$6^{th}$']
    string11    =  ['1st','2nd','3rd','4th','5th','6th']
    fig, axs = plt.subplots(nrows=6, ncols=1, sharex=False, sharey=False, figsize=(20,15))
    fig.subplots_adjust(hspace = .4, wspace=.2)
    data = np.arange(0,len(string11))    

    for ax, kk in zip(axs.ravel(),data):
        file_to_read = open(r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\ISOIS\_"+string1[kk]+"orbit.dat", "rb")
        isois = pickle.load(file_to_read)
        file_to_read.close()
        isois =isois.set_index('DateTime')

        isois.plot(ax=ax,x_compat=True,color='black',lw=0.5)

        ax.set_ylim([2e-2,20])
        ax.set_xlim([df['Starting Dates'][kk],df['Ending Dates'][kk]])
    
            #### Set labels ####
        #ax.set_ylabel(r'$Count-rate \ (s^{-1})$', fontsize=18)
        ax.set_xlabel('')
    
            #### Set title ####
        #ax.title.set_text(str(string21[kk])+'$ \ Perihelion$')
            #### Set scale ####
        ax.set_yscale('log')
        
            ### Set legend ###
        ax.legend(frameon=False,loc=1,fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=15, rotation='default')
        #ax.tick_params(axis='both', which='minor', labelsize=15, rotation='default')
        #ax.xaxis.set_major_formatter(md.DateFormatter('%Y:%m:%D:'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
        # set formatter
        ax.text(0.07, 0.82, string21[kk]+"$ \ Encounter$",fontsize=17, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
# 21)Plot Count rates ions


def plot_count_rates_electrons(string1):
    file_to_read = open(r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\PVI\_Start_End_dates.dat", "rb")
    df = pickle.load(file_to_read)
    file_to_read.close()
    import matplotlib.dates as md
    string21           = ['$1^{st}$','$2^{nd}$','$3^{rd}$','$4^{th}$','$5^{th}$','$6^{th}$']
    string11    =  ['1st','2nd','3rd','4th','5th','6th']
    fig, axs = plt.subplots(nrows=6, ncols=1, sharex=False, sharey=False, figsize=(20,15))
    fig.subplots_adjust(hspace = .4, wspace=.2)
    data = np.arange(0,len(string11))    

    for ax, kk in zip(axs.ravel(),data):
        file_to_read = open(r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\ISOIS\_electrons"+string1[kk]+"orbit.dat", "rb")
        isois = pickle.load(file_to_read)
        file_to_read.close()
        isois =isois.set_index('DateTime')

        isois.plot(ax=ax,x_compat=True,color='black',lw=0.5)

        ax.set_ylim([9e-1,20])
        ax.set_xlim([df['Starting Dates'][kk],df['Ending Dates'][kk]])
    
            #### Set labels ####
        #ax.set_ylabel(r'$Count-rate \ (s^{-1})$', fontsize=18)
        ax.set_xlabel('')
    
            #### Set title ####
        #ax.title.set_text(str(string21[kk])+'$ \ Perihelion$')
            #### Set scale ####
        ax.set_yscale('log')
        
            ### Set legend ###
        ax.legend(frameon=False,loc=1,fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=15, rotation='default')
        #ax.tick_params(axis='both', which='minor', labelsize=15, rotation='default')
        #ax.xaxis.set_major_formatter(md.DateFormatter('%Y:%m:%D:'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
        # set formatter
        ax.text(0.07, 0.82, string21[kk]+"$ \ Encounter$",fontsize=17, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


	
    ################################################################################################################    

# 18) Estimate PVI and save PVI

def estimate_pvi_HEP(B, ind1, ind2, tau, hours, string11, perih):
    import functions2 as fun
    B      = B[ind1:ind2]
    ar     = B.Br.values
    at     = B.Bt.values
    an     = B.Bn.values

    ###  Estimate cadence of DF ###
    lag     = (B.index[1]-B.index[0])/np.timedelta64(1,'s') 
    
    DBtotal = fun.func_DB(ar, at, an, tau)
    PVI     = fun.func_PVI(DBtotal, tau, lag, hours)
    PVI_df  = pd.DataFrame({'DateTime': B.index,
                                 'PVI': PVI})
    PVI_df  = PVI_df.set_index('DateTime')
        
    ### Store PVI dataframe ###
    file_to_store = open(r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\PVI\_"+string11[perih]+".dat", "wb") # sto trito bale 2 opws einai twra
    pickle.dump(PVI_df, file_to_store)
    file_to_store.close()
        
    ### Read Count rate and resample PVI accordingly ###
    file_to_read = open(r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\ISOIS\_"+string11[perih]+"orbit.dat", "rb")
    ISOIS = pickle.load(file_to_read)
    file_to_read.close()
    ISOIS = ISOIS.set_index('DateTime')
        
    new_index   = ISOIS.index.values
    PVI_df_new     = PVI_df.reindex(PVI_df.index.union(new_index)).interpolate(method='linear',how ='max').reindex(new_index)
    #PVI_df_new  = PVI_df.interpolate( how='max', fill_method='ffill').reindex(new_index)
    #PVI_df_new  = fun.interp(PVI_df, new_index)
    final_df    = pd.DataFrame({'DateTime': new_index,
                                     'PVI':PVI_df_new.values.T[0],
                                  'Counts': ISOIS['$Count-rate \ (s^{-1})$'].values})
    final_df = final_df.set_index('DateTime')

    
    file_to_store = open(r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\merged_PVI_ISOIS\_"+string11[perih]+"_orbit.dat", "wb") # sto trito bale 2 opws einai twra
    pickle.dump(final_df, file_to_store)
    file_to_store.close()


# 22) estimate_pvi_HEP_1sec
def estimate_pvi_HEP_1sec(B, ind1, ind2, tau, hours, string11, perih):
    import functions2 as fun
    B      = B[ind1:ind2]
    ar     = B.Br.values
    at     = B.Bt.values
    an     = B.Bn.values

    ###  Estimate cadence of DF ###
    lag     = (B.index[1]-B.index[0])/np.timedelta64(1,'s') 
    
    DBtotal = fun.func_DB(ar, at, an, tau)
    PVI     = fun.func_PVI(DBtotal, tau, lag, hours)
    PVI_df  = pd.DataFrame({'DateTime': B.index,
                                 'PVI': PVI})
    PVI_df  = PVI_df.set_index('DateTime')
        
    ### Store PVI dataframe ###
    file_to_store = open(r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\PVI\_"+string11[perih]+"_1sec.dat", "wb") # sto trito bale 2 opws einai twra
    pickle.dump(PVI_df, file_to_store)
    file_to_store.close()
        
    #### Read Count rate and resample PVI accordingly ###
    #file_to_read = open(r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\ISOIS\_"+string11[perih]+"orbit.dat", "rb")
    #ISOIS = pickle.load(file_to_read)
    #file_to_read.close()
    #ISOIS = ISOIS.set_index('DateTime')
        
    #new_index   = ISOIS.index.values
    #PVI_df_new     = PVI_df.reindex(PVI_df.index.union(new_index)).interpolate(method='linear',how ='max').reindex(new_index)
    #PVI_df_new  = PVI_df.interpolate( how='max', fill_method='ffill').reindex(new_index)
    #PVI_df_new  = fun.interp(PVI_df, new_index)
    #final_df    = pd.DataFrame({'DateTime': new_index,
     #                                'PVI':PVI_df_new.values.T[0],
     #                             'Counts': ISOIS['$Count-rate \ (s^{-1})$'].values})
    #final_df = final_df.set_index('DateTime')

    
    #file_to_store = open(r"C:\Users\nikos.000\PVI_HIGH_ENERGY_PARTICLES\data\merged_PVI_ISOIS\_"+string11[perih]+"_orbit.dat", "wb") # sto trito bale 2 opws einai twra
    #pickle.dump(final_df, file_to_store)
    #file_to_store.close()

