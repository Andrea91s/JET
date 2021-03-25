# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:04:10 2020

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression
from matplotlib import gridspec
#import random as rnd

#error on the linear regression coefficients ####
def errorFitb(asi, efit, inter, coeff):
    #return (((1/(len(asi)-2))* np.sum((asi-inter-coeff*efit)**2))**0.5) * ( (len(asi))          / (len(asi)*np.sum((efit)**2) - (np.sum(efit))**2)  )**0.5
    return (((1/(len(asi)))* np.sum((asi-inter-coeff*efit)**2))**0.5)  / ((np.sum((efit-np.mean(efit))**2))**0.5)

def errorFita(asi, efit, inter, coeff):
    #return (((1/(len(asi)-2))* np.sum((asi-inter-coeff*efit)**2))**0.5) * ( (np.sum((efit)**2)) / (len(asi)*np.sum((efit)**2) - (np.sum(efit))**2)  )**0.5
    return ((((1/(len(asi)))* np.sum((asi-inter-coeff*efit)**2))**0.5)  / ((np.sum((efit-np.mean(efit))**2))**0.5)) * ((np.sum(efit**2)/len(asi))**0.5)


##### POISSON ERROR ON NC COUNTS ####
def errcounts(l,area,los):
    return ((l**2)/(area**2))*(np.abs(los)**0.5)

def errorcoeff(we):
    return np.sqrt(1/np.sum(1/we**2))


def sigma(fact, avgfact):
    return np.sqrt((1/len(fact)) * np.sum((fact-avgfact)**2))

def interpolation(newtime, oldtime, array):
    return np.interp(newtime, oldtime, array)

zefit_big = []
refit_big = []
momentH_big = []
momentV_big = []
asimhf_big = [] 
asimvf_big = []
sigmaasimhf_big = []
sigmaasimvf_big = []
errasimh_big = []
errasimv_big = []
errmomh_big = []
errmomv_big = []

#
#acoeffH  =   np.array((-0.579,-0.559,-0.597,-0.542))
#acoeffHerr = np.array((0.003, 0.002, 0.002, 0.002))
#bcoeffH    = np.array((2.33,2.25,2.302,2.18))
#bcoeffHerr = np.array((0.01,0.01, 0.01,0.01))
#
#acoeffV    = np.array((5.29,5.8,4.5, 4.12))
#acoeffVerr = np.array((0.02,0.015,0.04,0.03))
#bcoeffV    = np.array((-1.697,-1.86,-1.432,-1.31))
#bcoeffVerr = np.array((0.01,0.005,0.01,0.01))
#
#
#
#
#acoeffHm  =   np.array((5.95,5.81,5.847,5.72))
#acoeffHmerr = np.array((0.015, 0.01, 0.01, 0.01))
#bcoeffHm    = np.array((-5.58,-5.1,-5.09,-4.65))
#bcoeffHmerr = np.array((0.05,0.03,0.03,0.03))
#
#acoeffVm    = np.array((5.59,4.82, 6.77, 7.68))
#acoeffVmerr = np.array((0.08,0.07,0.1,0.08))
#bcoeffVm    = np.array((3.03,3.27,2.63,2.34))
#bcoeffVmerr = np.array((0.03,0.02,0.04,0.02))
#
#overaHfit = np.mean(acoeffH)
#overbHfit = np.mean(bcoeffH)
#overaVfit = np.mean(acoeffV)
#overbVfit = np.mean(bcoeffV)
#
#overaHfitm = np.mean(acoeffHm)
#overbHfitm = np.mean(bcoeffHm)
#overaVfitm = np.mean(acoeffVm)
#overbVfitm = np.mean(bcoeffVm)



overaHfit = -0.539
overbHfit = 2.057
overaVfit = 2.61
overbVfit = -0.947

overaHfitm = 5.473
overbHfitm = -3.68
overaVfitm = 11.24
overbVfitm = 1.31
#
#shot=[ 84536, 84539, 84540, 84541, 84542, 84543, 84544, 84545, 84786,
#       84788, 84789, 84792, 84796, 84797, 84798, 85197, 85201, 85203,
#       85204, 85205, 85228, 85229, 85230, 85231, 85232, 85393, 85394,
#       85399, 85400, 85401, 85402, 85403, 86362, 86365, 86366, 86367,
#       86368, 86374, 86375, 86377, 86378, 86379, 86380, 86381, 86382,
#       86383, 86386, 86388, 86389, 86390, 86392, 86399, 86401, 86404,
#       86407, 86411, 86412, 86413, 86414, 86416, 86417, 86419, 86421,
#       86435, 86436, 86437, 86439, 86441, 86444, 86445, 86598, 86599,
#       86600, 86602, 86603, 86604, 86605, 86606, 86658, 86659, 86660,
#       86661, 86662, 86663, 86664, 86982, 86983, 86985, 86986, 86987, 
#       86988, 86989, 86991, 86992, 86993, 86994, 87140, 87142, 87143, 
#       87144, 87160, 87161, 87162, 87164, 87228, 87231, 87232, 87233, 
#       87240, 87243, 87244, 87246, 87247, 87248, 87249, 87250, 87251, 
#       87252, 87253, 87267, 87268, 87271, 87273, 87276, 87277, 87278,
#       87280, 92393, 92394, 92395, 92398, 92399, 94256, 94257, 94259,
#       94261, 94262, 94263, 94264, 94265, 94266, 94268, 94270, 94665, 
#       94666, 94667, 94669, 94670, 94671, 94672, 94674, 94675, 94676, 
#       94677, 94678, 94679, 94680, 94681, 94682, 94683, 94684]

shot = [96435];countsleft=200 ;countsright=200; filt=21
for w in range(0, len(shot)):
    plt.close('all')
    print(shot[w])
    
    time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCtime_'+ str(shot[w])+'.txt')
    f = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCdata_'+ str(shot[w])+'.txt')
    refita = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/REFIT_'+ str(shot[w])+'.txt')
    zefita = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ZEFIT_'+ str(shot[w])+'.txt')
    time_maga = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/time_mag_'+ str(shot[w])+'.txt')
    nbi_time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/nbi_time'+ str(shot[w])+'.txt')
    nbi = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/nbi'+ str(shot[w])+'.txt')
    print('The integration time of the EFIT is', round(float(np.diff(time_maga)[0]*1000)),'ms')    
        

    counts_old = np.reshape(f,(len(time),20))   #NEW NEUTRON CAMERA
    counts = np.zeros((len(time_maga),20))
    
    for i in range(len(counts_old[1])):
        counts[:,i] = round(float(np.diff(time_maga)[0]*1000))*interpolation(time_maga,time,counts_old[:,i])/round(float(np.diff(time)[0]*1000))
   

# Here I select one the entries with number of counts greater than a certain number, both for left ansd right time limit

    timeindex = np.sum(counts,1)

    idxa = (np.where((timeindex>countsleft)))
    idxb = (np.where((timeindex>countsright)))


    a = time_maga[idxa] 
    b = time_maga[idxb]
    idxnbi = np.where((nbi>1e6))
    if a[0]>nbi_time[idxnbi][0]:
        t1 = a[0] 
    else:
        t1 = nbi_time[idxnbi][0]
        
    if b[-1]<nbi_time[idxnbi][-1]:
        t2 = b[-1] 
    else:            
        t2 = nbi_time[idxnbi][-1]
   
    t1=47.5
    t2=49.5


    

#Here I define the length and the area of collimators

    Len   = np.array((  1.615991444 ,  1.581272864 , 1.552087292 ,  1.532928955 , 1.522378565 ,  1.521778565 ,  1.533228955 , 1.553687292 ,  1.581572864 ,  1.617691444 ,  1.518333124 ,  1.515424142 ,  1.511876256 ,  1.510795114 ,  1.5093 ,  1.510695237 ,  1.514264589 ,  1.514830584 ,   1.518823781))
    radii = np.array((        1.050 ,        1.050 ,       1.050 ,        1.050 ,       1.050 ,        1.050 ,        1.050 ,       1.050 ,        1.050 ,        1.050 ,        0.500 ,        0.600 ,        0.750 ,         0.85 ,   1.050 ,        1.050 ,        1.050 ,        1.050 ,         1.050))/100.0 
    areas = np.pi * (radii**2) 

 #the horizontal camera has  10 channels (1-10)
# the vertical camera has 9 channels (11-19)

    Hchannel = np.linspace(1,10,10)
    Vchannel = np.linspace(11,19,9)
    Ha = counts[0:-1, 0:10]
    Va = counts[0:-1, 10:19]
    


# 
 # HERE I CONVERTED FROM NEUTRON EMISSION RATE (n/s) TO NEUTRON EMISSIVITY (n/s/m^2)    
    H = np.multiply(Ha,Len[0:10]**2) / (areas[0:10]**2)
    V = np.multiply(Va,Len[10:19]**2) / (areas[10:19]**2)   
    
  
    # HERE I SELECT ONLY THE ELEMENTS IN THE WANTED TIME WINDOW BECAUSE THERE ARE EMPTY SPACES

    idx = (np.where((time_maga>t1) & (time_maga<t2)))
    t = time_maga[idx]
    time_mag = time_maga[idx]
    refit = refita[idx]
    zefit = zefita[idx]
    H = H[idx]
    V = V[idx]
    
    Hfake = np.multiply(H,areas[0:10]**2) /   (Len[0:10]**2)
    Vfake = np.multiply(V,areas[10:19]**2) / (Len[10:19]**2)  
    H=Hfake
    V=Vfake

    #H[:,8]=0
    #H[:,9]=0


    momentH = np.zeros(len(t))
    momentV = np.zeros(len(t))
    errmomh = np.zeros(len(t))
    errmomv = np.zeros(len(t))
    numbercounts = np.zeros(len(t))
    all_zeros = np.zeros(len(t))

#IF THERE ARE ARRAYS WITH ALL ZERO ELEMENTS I FILL THEM WITH THE PREVIOUS NUMBER OF COUNTS

    for i in range(0, len(momentH)):
        all_zeros[i] = not np.any(H[i,:])
        if all_zeros[i] == True:
            H[i,:] = 100*np.ones((len(Hchannel)))
            #H[i,:] = H[i-1,:]

    for i in range(0, len(momentV)):
        all_zeros[i] = not np.any(V[i,:])
        if all_zeros[i] == True:
            V[i,:] = 100*np.ones((len(Vchannel)))
            #V[i,:] = V[i-1,:]
        
# HERE I CALCULATE THE FIRST MOMENT        

    for i in range(0, len(momentH)):
        momentH[i] = np.average(Hchannel, weights = H[i,:])
        momentV[i] = np.average(Vchannel, weights = V[i,:])
        errmomh[i] = 1/np.sqrt(np.sum(H[i,:]))
        errmomv[i] = 1/np.sqrt(np.sum(V[i,:]))
        numbercounts[i]  = np.sum(H[i,:])  
    
     

# THE OBTAINED SIGNAL IS CRAP SO I SMOOTH IT WITH A SAVITZKY-GOLAY FILTER

    filtr=signal.savgol_filter(momentV, filt, 3)
    filtz=signal.savgol_filter(momentH, filt, 3)
    #filtr=momentV
    #filtz=momentH

    a1 = np.zeros((len(t)))
    a2 = np.zeros((len(t)))
    a4 = np.zeros((len(t)))
    a5 = np.zeros((len(t)))
    
#this is calculated according to https://iopscience.iop.org/article/10.1088/1748-0221/14/09/C09001/pdf

    for i in range(0, len(t)):
        a1[i] = H[i,0] + H[i,1] + H[i,2] + H[i,3] 
        a2[i] = H[i,4] + H[i,5] + H[i,6] + H[i,7] + H[i,8] + H[i,9]
        a4[i] = V[i,0] + V[i,1] + V[i,2] + V[i,3] + V[i,4]/2 
        a5[i] = V[i,4]/2 + V[i,5] + V[i,6] + V[i,7] + V[i,8] 
   
    asimh = (a1 - a2) / (a1 + a2)
    asimv = (a4 - a5) / (a4 + a5)    

    errasimh = ((2*np.sqrt(a1*a2))/(np.sqrt((a1+a2)**3)))
    errasimv = ((2*np.sqrt(a5*a4))/(np.sqrt((a4+a5)**3)))
    
# AGAIN FILTERING 

    asimhf=signal.savgol_filter(asimh, filt, 3)
    asimvf=signal.savgol_filter(asimv, filt, 3)
    #asimhf=asimh
    #asimvf=asimv
#
#    
    refit_new = refit
    zefit_new = zefit
#    
    zefit_big = np.append(zefit_big, zefit_new)
    refit_big = np.append(refit_big, refit_new)
    momentH_big = np.append(momentH_big, momentH)
    momentV_big = np.append(momentV_big, momentV)
    asimhf_big = np.append(asimhf_big, asimhf)
    asimvf_big = np.append(asimvf_big, asimvf)
    errasimh_big = np.append(errasimh_big, errasimh)
    errasimv_big = np.append(errasimv_big, errasimv)
    errmomh_big = np.append(errmomh_big, errmomh)
    errmomv_big = np.append(errmomv_big, errmomv)
    
    
      # HERE I AM PLOTTING THE ASYMMETRY WITH THE NBI TRACES IN ORDER TO CHECK WEIRD BEHAVIOUR DUE TO MISSING NBI POWER
    
    plotH = Hfake.transpose().copy()
    plotH[np.where(plotH<1)] = np.nan
    
    plotV = Vfake.transpose().copy()
    plotV[np.where(plotV<1)] = np.nan 
        
    [T,CHH] = np.meshgrid(t, np.linspace(1,10,10)) 

    # HERE I AM PLOTTING THE ASYMMETRY WITH THE NBI TRACES IN ORDER TO CHECK WEIRD BEHAVIOUR DUE TO MISSING NBI POWER
    
    plt.figure(200,figsize = (17, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1]) 
    
    ax0 = plt.subplot(gs[0])
    plt.contourf(T,CHH, plotH, cmap = 'jet')
    plt.xlim(t1, t2)
    plt.ylabel('channel', fontsize=20)
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.yticks(fontsize=16)
    
    ax1 = plt.subplot(gs[1], sharex = ax0)
    plt.plot(nbi_time,nbi/1e6, 'k')
    plt.ylabel('NBI  (MW)', fontsize=20)
    plt.xlim(t1, t2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.yticks(fontsize=16)
    
    ax2=plt.subplot(gs[2], sharex = ax0)   
    #plt.plot(t, (asimhf-fitaH.intercept_)/fitaH.coef_[0], 'r', linewidth=2,label = 'Asym H', zorder=10)
    plt.plot(t, (asimhf-(overaHfit))/overbHfit, 'r', linewidth=2,label = 'Asym H', zorder=15)
    plt.plot(t, (filtz-overaHfitm)/overbHfitm, 'b', linewidth=2,label = 'Moment', zorder=10)
    plt.plot(t, zefit_new,'k', linewidth=2,label = 'Z EFIT') 
    plt.plot(t, (asimhf-(overaHfit-0.05))/(overbHfit+0.1), 'g--', linewidth=2,label = 'lower err', zorder=5)
    plt.plot(t, (asimhf-(overaHfit+0.05))/(overbHfit-0.1), 'g--', linewidth=2,label = 'upper err', zorder=5)      
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('Z (m)', fontsize=20)
    plt.legend(['Asymmetry H','Moment','Z EFIT'],fontsize=8,frameon=False)
    plt.xticks(np.linspace(t1,t2,6),fontsize=18)
    plt.yticks(np.linspace(0.19, 0.35,9), fontsize=18)
    plt.xlim(t1, t2)
    plt.ylim(0.18, 0.36)   
    plt.setp(ax2.get_xticklabels(), visible=False)    
    
    
    
    plt.subplot(gs[3], sharex = ax0)   
    #plt.plot(t, 100*(((asimhf-fitaH.intercept_)/fitaH.coef_[0])-(zefit_new)), 'r', linewidth=2,label = 'asym', zorder=15)  
    plt.plot(t, 100*(((asimhf-overaHfit)/overbHfit)-(zefit_new)), 'r--', linewidth=2,label = 'total asym', zorder=14)
    #plt.plot(t, 100*(((filtz-fitmH.intercept_)/fitmH.coef_[0])-(zefit_new)), 'b', linewidth=2,label = 'single asym', zorder=13) 
    plt.plot(t, 100*(((filtz-overaHfitm)/overbHfitm)-(zefit_new)), 'b--', linewidth=2,label = 'moment', zorder=12)
    plt.plot(t, 1.5*np.ones((len(t))), 'k')
    plt.plot(t, -1.5*np.ones((len(t))), 'k')
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('diff (cm)', fontsize=20)
    plt.xticks(np.linspace(t1,t2,6),fontsize=16)
    plt.yticks(np.linspace(-10,10,7), fontsize=16)
    #plt.legend(['Asym H','Asym overall H','Moment', 'Moment overall'],fontsize=8,frameon=False, ncol=2)
    plt.legend(['Asym overall H','Moment overall'],fontsize=8,frameon=False, ncol=2)
    plt.xlim(t1, t2)      
    plt.subplots_adjust(hspace=.0)
    plt.ylim(-5, 5)
    plt.subplots_adjust(hspace=.0)

    #plt.savefig('zmissingNBI_'+ str(ind) + 'and' + str(ind+10) + '_' + str(shot[w]) + '_' + str(filt) + '_H.png')
    plt.savefig('zeNBI_'+ str(shot[w]) + '__H.png')

    
    [T,CHV] = np.meshgrid(t, np.linspace(11,19,9))    

    
    plt.figure(300,figsize = (17, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])  
    
    ax0 = plt.subplot(gs[0])
    plt.contourf(T, CHV, plotV, cmap='jet')
    plt.xlim(t1, t2)
    plt.ylabel('channel', fontsize=20)
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.yticks(fontsize=16)
    
    ax1 = plt.subplot(gs[1], sharex = ax0)
    plt.plot(nbi_time,nbi/1e6, 'k')
    plt.ylabel('NBI (MW)', fontsize=20)
    plt.xlim(t1, t2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.yticks(fontsize=16)
    
    ax2=plt.subplot(gs[2], sharex = ax0)   
    #plt.plot(t, (asimvf-fitaV.intercept_)/fitaV.coef_[0], 'r', linewidth=2,label = 'Asym V', zorder=10)
    plt.plot(t, (asimvf-overaVfit)/overbVfit, 'r', linewidth=2,label = 'Asym V', zorder=15)
    plt.plot(t, (filtr-overaVfitm)/overbVfitm, 'b', linewidth=2,label = 'Moment', zorder=10)
    #plt.plot(t, (filtr-fitmV.intercept_)/fitmV.coef_[0], 'b', linewidth=2,label = 'Moment', zorder=10)
    plt.plot(t, refit_new,'k', linewidth=2,label = 'R EFIT')
    plt.plot(t, (asimvf-(overaVfit-0.4))/(overbVfit+0.1), 'g--', linewidth=2,label = 'lower err', zorder=5)
    plt.plot(t, (asimvf-(overaVfit+0.4))/(overbVfit-0.1), 'g--', linewidth=2,label = 'upper err', zorder=5)
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('R (m)', fontsize=20)
    #plt.legend(['Asymmetry V','Asymmetry V overall','R EFIT'],fontsize=16,frameon=False)
    plt.legend(['Asymmetry V','Moment','R EFIT'],fontsize=16,frameon=False)
    plt.xticks(np.linspace(t1,t2,6),fontsize=16)
    plt.yticks(np.linspace(2.90,3.20,7), fontsize=16)
    plt.xlim(t1, t2)
    plt.ylim(2.85, 3.25)        
    plt.subplots_adjust(hspace=.0)
    plt.setp(ax2.get_xticklabels(), visible=False)

    
    plt.subplot(gs[3], sharex = ax0)   
    #plt.plot(t, 100*(((asimvf-fitaV.intercept_)/fitaV.coef_[0])-(refit_new)), 'r', linewidth=2,label = 'asym', zorder=15)  
    plt.plot(t, 100*(((asimvf-overaVfit)/overbVfit)-(refit_new)), 'r--', linewidth=2,label = 'total asym', zorder=14)
    #plt.plot(t, 100*(((filtr-fitmV.intercept_)/fitmV.coef_[0])-(refit_new)), 'b', linewidth=2,label = 'single asym', zorder=13) 
    plt.plot(t, 100*(((filtr-overaVfitm)/overbVfitm)-(refit_new)), 'b--', linewidth=2,label = 'moment', zorder=12)
    plt.plot(t, 1.5*np.ones((len(t))), 'k')
    plt.plot(t, -1.5*np.ones((len(t))), 'k')
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('diff (cm)', fontsize=20)
    #plt.legend(['Asym V','Asym overall V','Moment', 'Moment overall'],fontsize=8,frameon=False, ncol=2)
    plt.legend(['Asym overall V','Moment overall'],fontsize=8,frameon=False, ncol=2)
    plt.xticks(np.linspace(t1,t2,6),fontsize=16)
    plt.yticks(np.linspace(-10,10,7), fontsize=16)
    plt.xlim(t1, t2)   
    plt.subplots_adjust(hspace=.0)
    plt.ylim(-5, 5)
    
    
    
    #plt.savefig('zmissingNBI_'+ str(ind) + 'and' + str(ind+10) + '_' + str(shot[w]) + '_' + str(filt) + '_V.png')
    plt.savefig('zeNBI_'+ str(shot[w]) + '__V.png')
    


    ##### THOSE ARE FOR THE ASYMMETRY #######
#indexh = (np.where((errasimh_big==0)))
#indexv = (np.where((errasimv_big==0)))
#
#errasimh_big[indexh] = np.mean(errasimh_big)
#errasimv_big[indexv] = np.mean(errasimv_big)
#
#    
#fitaH = LinearRegression().fit(zefit_big.reshape((-1, 1)), asimhf_big, sample_weight = 1/errasimh_big**2)
#fitaV = LinearRegression().fit(refit_big.reshape((-1, 1)), asimvf_big, sample_weight = 1/errasimv_big**2)
#
#fitmH = LinearRegression().fit(zefit_big.reshape((-1, 1)), momentH_big, sample_weight = 1/errmomh_big**2)
#fitmV = LinearRegression().fit(refit_big.reshape((-1, 1)), momentV_big, sample_weight = 1/errmomv_big**2)
#
#r_sqmH = fitmH.score(zefit_big.reshape((-1, 1)), momentH_big)
#r_sqmV = fitmV.score(refit_big.reshape((-1, 1)), momentV_big)
#r_sqaH = fitaH.score(zefit_big.reshape((-1, 1)), asimhf_big)
#r_sqaV = fitaV.score(refit_big.reshape((-1, 1)), asimvf_big)
#
#Z = np.linspace(np.min(zefit_big), np.max(zefit_big), 1000)
#R = np.linspace(np.min(refit_big), np.max(refit_big), 1000)

##### PLOT THAT SHOWS THE OVERALL LINEAR FIT  #######
#
#np.savetxt('S' + str(scenario)+ '_ZEFIT.txt',zefit_big)
#np.savetxt('S' + str(scenario)+ '_REFIT.txt',refit_big)
#np.savetxt('S' + str(scenario)+ '_HASIM.txt',asimhf_big)
#np.savetxt('S' + str(scenario)+ '_VASIM.txt',asimvf_big)

#
#plt.figure(400,figsize = (24, 17))
##plt.suptitle(shot, fontsize=26)
#
#plt.subplot(2,2,1)
#plt.text(np.max(zefit_big), np.max(momentH_big), '$R^2$ = (%.3f)'%r_sqmH,
#         verticalalignment='top', horizontalalignment='right', fontsize=25) 
#plt.scatter(zefit_big, momentH_big, c='b', s=10)
#plt.plot(Z, fitmH.intercept_+Z*fitmH.coef_, c='r', linewidth=3)
#plt.xlabel('Z (m)', fontsize=20)
#plt.ylabel('Moment H', fontsize=20)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#   
#plt.subplot(2,2,2)
#plt.text(np.max(refit_big), np.max(momentV_big), '$R^2$ = (%.3f)'%r_sqmV,
#         verticalalignment='top', horizontalalignment='right', fontsize=25) 
#plt.scatter(refit_big, momentV_big, c='r', s=10)
#plt.plot(R, fitmV.intercept_+R*fitmV.coef_, c='b', linewidth=3)
#plt.xlabel('R (m)', fontsize=20)
#plt.ylabel('Moment V', fontsize=20)    
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#    
#plt.subplot(2,2,3)
#plt.plot(Z, fitaH.intercept_+Z*fitaH.coef_, c='r', linewidth=3, zorder=10)
##plt.plot(Z, (fitaH.intercept_+erraH) + Z*fitaH.coef_, 'r--', linewidth=2)
##plt.plot(Z, (fitaH.intercept_-erraH) + Z*fitaH.coef_, 'r--', linewidth=2)
#plt.text(np.max(zefit_big), np.min(asimhf_big), '$R^2$ = (%.3f)'%r_sqaH,
#         verticalalignment='bottom', horizontalalignment='right', fontsize=25) 
#plt.scatter(zefit_big, asimhf_big, c='b', s=10)
##plt.errorbar(zefit_big, asimhf_big, yerr = errasimh_big, fmt='o', c='k', zorder=0)
#plt.xlabel('Z (m)', fontsize=20)
#plt.ylabel('Asymmetry H', fontsize=20)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#    
#plt.subplot(2,2,4)
#plt.text(np.max(refit_big), np.max(asimvf_big), '$R^2$ = (%.3f)'%r_sqaV,
#         verticalalignment='top', horizontalalignment='right', fontsize=25) 
#plt.scatter(refit_big, asimvf_big, c='r', s=10)
##plt.errorbar(refit_big, asimvf_big, yerr = errasimv_big, fmt='o', c='k', zorder=0)
#plt.plot(R, fitaV.intercept_        + R*fitaV.coef_, c='b', linewidth=3, zorder=10)
##plt.plot(R, (fitaV.intercept_+erraV)+ R*fitaV.coef_, 'b--', linewidth=2)
##plt.plot(R, (fitaV.intercept_-erraV)+ R*fitaV.coef_, 'b--', linewidth=2)
#plt.xlabel('R (m)', fontsize=20)
#plt.ylabel('Asymmetry V', fontsize=20)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)


#np.savetxt('/media/andrea/My Passport Ultra/REFITM.txt', refit_big)
#np.savetxt('/media/andrea/My Passport Ultra/ZEFITM.txt', zefit_big)
#np.savetxt('/media/andrea/My Passport Ultra/ASYV.txt', asimvf_big)
#np.savetxt('/media/andrea/My Passport Ultra/ASYH.txt', asimhf_big)
#np.savetxt('/media/andrea/My Passport Ultra/MOMV.txt', momentV_big)
#np.savetxt('/media/andrea/My Passport Ultra/MOMH.txt', momentH_big)

#plt.savefig('zzzzzzzzza' + str(shot) + '_' + str(filt) + '.png')

#plt.close('all')

erraH = errorFita(asimhf_big, zefit_big, fitaH.intercept_ , fitaH.coef_[0])
errbH = errorFitb(asimhf_big, zefit_big, fitaH.intercept_ , fitaH.coef_[0])
erraV = errorFita(asimvf_big, refit_big, fitaV.intercept_ , fitaV.coef_[0])
errbV = errorFitb(asimvf_big, refit_big, fitaV.intercept_ , fitaV.coef_[0])


erramH = errorFita(momentH_big, zefit_big, fitmH.intercept_ , fitmH.coef_[0])
errbmH = errorFitb(momentH_big, zefit_big, fitmH.intercept_ , fitmH.coef_[0])
erramV = errorFita(momentV_big, refit_big, fitmV.intercept_ , fitmV.coef_[0])
errbmV = errorFitb(momentV_big, refit_big, fitmV.intercept_ , fitmV.coef_[0])




#print('The a coefficient for the asymmetry HC is = ', round(fitaH.intercept_,5), '+/-' , round(erraH,5))
#print('The b coefficient for the asymmetry HC is = ',  round(fitaH.coef_[0],5), '+/-' , round(errbH,5))
#
#print('The a coefficient for the asymmetry VC is = ',  round(fitaV.intercept_,5), '+/-' , round(erraV,5))
#print('The b coefficient for the asymmetry VC is = ',    round(fitaV.coef_[0],5), '+/-' , round(errbV,5))
#print('')
#print('################################################################################')
#print('################################################################################')
#print('')    
#print('The a coefficient for the moment HC is = ', round(fitmH.intercept_,5), '+/-' , round(erramH,5))
#print('The b coefficient for the moment HC is = ',  round(fitmH.coef_[0],5), '+/-' , round(errbmH,5))
#
#print('The a coefficient for the moment VC is = ',  round(fitmV.intercept_,5), '+/-' , round(erramV,5))
#print('The b coefficient for the moment VC is = ',    round(fitmV.coef_[0],5), '+/-' , round(errbmV,5))


#plt.close('all')


xfitR = refit_new
yfitR = (asimvf-overaVfit)/overbVfit

xfitZ = zefit_new
yfitZ = (asimhf-overaHfit)/overbHfit

deltaR = (asimvf-overaVfit)/overbVfit -refit_new
deltaZ = (asimhf-overaHfit)/overbHfit -zefit_new

stdR = np.sum((deltaR-np.mean(deltaR))**2)/len(deltaR)
stdZ = np.sum((deltaZ-np.mean(deltaZ))**2)/len(deltaZ)

#print('Delta R is ',round(np.mean(deltaR)*100,5) ,' +- ',round(stdR*100,5),' cm')
print('Delta Z is ',round(np.mean(deltaZ)*100,5) ,' +- ',round(stdZ*100,5),' cm')

#
#kR = LinearRegression().fit(xfitR.reshape((-1, 1)), yfitR,sample_weight=(1/(0.015*np.ones(len(xfitR))))**2)
#kZ = LinearRegression().fit(xfitZ.reshape((-1, 1)), yfitZ,sample_weight=(1/(0.015*np.ones(len(xfitZ))))**2)
#
#
#print('Zinter=', round(kZ.intercept_,5))
#print('Zcoef=', kZ.coef_,5)
#print('%%%%%%%%%%%%%%%%%%')
#print('Rinter=', round(kR.intercept_,5))
#print('Rcoef=', kR.coef_,5)

#plt.figure(1)
#plt.scatter(xfitZ, yfitZ)
#plt.plot(np.linspace(min(xfitZ),max(xfitZ),100), np.linspace(min(xfitZ),max(xfitZ),100)*kZ.coef_+kZ.intercept_)
#
#
#
#plt.figure(2)
#plt.scatter(xfitR, yfitR)
#plt.plot(np.linspace(min(xfitR),max(xfitR),100), np.linspace(min(xfitR),max(xfitR),100)*kR.coef_+kR.intercept_)
#


