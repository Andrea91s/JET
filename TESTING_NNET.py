# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:04:10 2020

@author: andrea
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Input
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import get_custom_objects
from tensorflow.python.keras.models import load_model
from scipy import stats
import random as rnd
from matplotlib import gridspec
from scipy import signal

def pure_linear(x):
    return x

def f(x):
    return np.int(x)
f2 = np.vectorize(f)

get_custom_objects().update({'pure_linear': pure_linear})




modelR = load_model("Model_from_counts_to_position_V_all.h5")
modelZ = load_model("Model_from_counts_to_position_H_all.h5")


#shot = [85186]  # THIS IS THE NICE ONE WITH THE DISPLACEMENT
#filt = 95 ; countsleft=100 ; countsright=100
shot = [84540] ; scenario=1
filt=47; countsleft=200 ; countsright=200

for w in range(0, len(shot)):
    #plt.close('all')
    print(shot[w])
    
    time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCtime_'+ str(shot[w])+'.txt')
    f = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCdata_'+ str(shot[w])+'.txt')
    refita  = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/REFIT_'+ str(shot[w])+'.txt')
    zefita = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ZEFIT_'+ str(shot[w])+'.txt')    
    time_maga = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/time_mag_'+ str(shot[w])+'.txt')

    counts = np.reshape(f,(len(time),20))/100   #NEW NEUTRON CAMERA
    timeindex = np.sum(counts,1)

    idxa = (np.where((timeindex>countsleft)))
    idxb = (np.where((timeindex>countsright)))
    a = time[idxa] 
    b = time[idxb]

    t1 = a[0] 
    t2 = b[-1] 
#    t1 = 46.5   ##for 92398
#    t2 = 49.03  ##for 92398
  
    Len   = np.array((  1.615991444 ,  1.581272864 , 1.552087292 ,  1.532928955 , 1.522378565 ,  1.521778565 ,  1.533228955 , 1.553687292 ,  1.581572864 ,  1.617691444 ,  1.518333124 ,  1.515424142 ,  1.511876256 ,  1.510795114 ,  1.5093 ,  1.510695237 ,  1.514264589 ,  1.514830584 ,   1.518823781))
    radii = np.array((        1.050 ,        1.050 ,       1.050 ,        1.050 ,       1.050 ,        1.050 ,        1.050 ,       1.050 ,        1.050 ,        1.050 ,        0.500 ,        0.600 ,        0.750 ,         0.85 ,   1.050 ,        1.050 ,        1.050 ,        1.050 ,         1.050))/100.0 
    areas = np.pi * (radii**2) 

 #the horizontal camera has  10 channels (1-10)
# the vertical camera has 9 channels (11-19)
    Ha = np.zeros((len(time),10))
    Va = np.zeros((len(time),9))
    Hchannel = np.linspace(1,10,10)
    Vchannel = np.linspace(11,19,9)
    Ha = counts[0:-1, 0:10]
    Va = counts[0:-1, 10:19]
    
    Ha[:,8]=1e-25
    Va[:,8]=1e-25
 
 # HERE I CONVERTED FROM NEUTRON EMISSION RATE (n/s) TO NEUTRON EMISSIVITY (n/s/m^2)    
    H = np.multiply(Ha,Len[0:10]**2) / (areas[0:10]**2)
    V = np.multiply(Va,Len[10:19]**2) / (areas[10:19]**2)   
        
  
    # HERE I SELECT ONLY THE ELEMENTS IN THE WANTED TIME WINDOW BECAUSE THERE ARE EMPTY SPACES

    idx = (np.where((time>t1) & (time<t2)))
    t = time[idx]
    idx_efit = (np.where((time_maga>t1) & (time_maga<t2)))
    time_mag = time_maga[idx_efit]
    refit = refita[idx_efit]
    zefit = zefita[idx_efit]
    H = H[idx]
    V = V[idx]

    
    
    
test_Hcounts_norm = stats.zscore(H)
test_Hcounts_mean = np.mean(H)
test_Hcounts_std = np.std(H, ddof=1)

test_Vcounts_norm = stats.zscore(V)
test_Vcounts_mean = np.mean(V)
test_Vcounts_std = np.std(V, ddof=1)
#
target_efit_R_norm = stats.zscore(refit)
target_efit_R_mean = np.mean(refit)
target_efit_R_std = np.std(refit, ddof=1)

target_efit_Z_norm = stats.zscore(zefit)
target_efit_Z_mean = np.mean(zefit)
target_efit_Z_std = np.std(zefit, ddof=1)
#
predR= modelR.predict(test_Vcounts_norm)
predZ= modelZ.predict(test_Hcounts_norm)


predaR = predR*target_efit_R_std+target_efit_R_mean
predaZ = predZ*target_efit_Z_std+target_efit_Z_mean

#
predR = signal.savgol_filter(np.ravel(predaR), filt, 3)
predZ = signal.savgol_filter(np.ravel(predaZ),filt, 3)

#asymR = np.loadtxt('92398_VR.txt')
#asymZ = np.loadtxt('92398_HZ.txt')

#asymR = np.loadtxt('85186_VR.txt')
#asymZ = np.loadtxt('85186_HZ.txt')


plt.figure(400,figsize = (17,10))
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1,1,1]) 

ax0 = plt.subplot(gs[0])
plt.plot(t,predR,'r')
#plt.plot(t,asymR,'g')
plt.plot(time_mag,refit,'k')
plt.ylabel('R (m)')
#plt.legend(['NNET','Asimmetry','EFIT'])
plt.legend(['NNET','EFIT'])
plt.xlim(t1, t2)
plt.ylim(2.85, 3.25)  
plt.show()
plt.setp(ax0.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.0)

ax1 = plt.subplot(gs[1])
plt.plot(t,predZ,'b')
#plt.plot(t,asymZ,'g')
plt.plot(time_mag,zefit,'k')
plt.ylabel('Z (m)')
plt.legend(['NNET','EFIT'])
plt.xlim(t1, t2)
plt.ylim(0.18, 0.36)
plt.show()
plt.setp(ax0.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.0)

refit = np.interp(t, time_mag, refit)
zefit = np.interp(t, time_mag, zefit)

ax0 = plt.subplot(gs[2])
plt.plot(t,100*(predR-refit),'r')
#plt.plot(t,100*(predR-asymR),'g')
plt.plot(np.linspace(min(t), max(t), 100),np.zeros((100,1)),'k--')
plt.ylabel(' diff R (cm)')
plt.xlim(t1, t2)
plt.ylim(-5, 5) 
plt.show()
plt.setp(ax0.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.0)

ax1 = plt.subplot(gs[3])
plt.plot(t,100*(predZ-zefit),'b')
#plt.plot(t,100*(predZ-asymZ),'g')
plt.plot(np.linspace(min(t), max(t), 100),np.zeros((100,1)),'k--')
plt.ylabel(' diff Z (cm)')
plt.xlabel('time (s)')
plt.xlim(t1, t2)
plt.ylim(-5, 5) 
plt.show()
plt.setp(ax0.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.0)