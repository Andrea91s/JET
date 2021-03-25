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
import random as rnd

#error on the linear regression coefficients ####
def errorFitb(asi, efit, inter, coeff):
    return (((1/(len(asi)-2))* np.sum((asi-inter-coeff*efit)**2))**0.5) * ( (len(asi)) / (len(asi)*np.sum((efit)**2) - (np.sum(efit))**2)  )**0.5

def errorFita(asi, efit, inter, coeff):
    return (((1/(len(asi)-2))* np.sum((asi-inter-coeff*efit)**2))**0.5)     * ( (np.sum((efit)**2)) / (len(asi)*np.sum((efit)**2) - (np.sum(efit))**2)  )**0.5



##### POISSON ERROR ON NC COUNTS ####
def errcounts(l,area,los):
    return ((l**2)/(area**2))*(np.abs(los)**0.5)

def errorcoeff(we):
    return np.sqrt(1/np.sum(1/we**2))


def sigma(fact, avgfact):
    return np.sqrt((1/len(fact)) * np.sum((fact-avgfact)**2)     )


def random_missing(counts):
    rnd.seed()
    ir = [rnd.randint(0, 19) for p in range(0, len(counts))]
    w = np.ones((len(counts), 20))
    for n in range(0,len(counts)):
        w[n, ir[n]] = 0
    return counts*w

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
ABC = []



########### I FEEL LIKE IT IS TOO SOON TO INCLUDE THOSE SINCE FUNNY THINGS ARE GOING ON  ###########
#shot = [86658, 86659,86660,86661,86662,86663,86664,86667,86668,86669,86670,86671,86672,86673,86674,86675,86676,86677,86678,86679] 
#filt=95 #countsleft>1000 #countsright>1000


#shot = [94665,94666,94667,94669,94670,94671,94672,94674,94675,94676,94677,94678,94679,94680,94681,94682,94683,94684] 
#filt=85
######################
#shot = [61167]  # THIS IS THE NICE ONE WITH THE DISPLACEMENT
#filt = 21

#shot = [85186]
#filt = 95



#shot = [84536, 84539, 84540, 84541,84542,84543,84544,84545,84786,84787,84788,84789,84792,84793,84794,84795,84796,84797,84798,85197, 85201, 85203, 85204, 85205,
#        85393, 85394, 85398, 85399] 
#filt = 47

shot = [92398]  # THIS IS THE NICE ONE WITH THE DISPLACEMENT
filt = 15 ; countsleft=100 ; countsright=100

############ THOSE ARE GREAT ##########
#shot = [84536,84539, 84540, 84541,84542,84543,84544,84545] 
#filt=47 #countsleft>200 #countsright>200
#shot = [84786,84787,84788,84789,84792,84793,84794,84795,84796,84797,84798] 
#filt=45 #countsleft>200 #countsright>200
#shot = [84802,84803, 84804, 84806, 84808] 
#filt=85 #countsleft>500 #countsright>2000
#shot = [85197, 85201, 85203, 85204, 85205] 
#filt=21 #countsleft>200 #countsright>2000
#shot = [85393, 85394, 85398, 85399] 
#filt=15 #countsleft>200 #countsright>500
#shot = [92393, 92394, 92395, 92398, 92399]
#filt = 45 #countsleft>200 #countsright>5800
#shot=[94256,94257,94259,94261,94262,94263,94264,94265,94266,94268,94270]
#filt =85 #countsleft>50 #countsright>50

############ THOSE ARE GREAT ##########



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
    
    #I DIVIDED BY 100 because the integration time is 10 ms
    counts = np.reshape(f,(len(time),20))/100   #NEW NEUTRON CAMERA
    #counts = np.reshape(f,(len(time),19))/100  #OLD NEUTRON CAMERA

# Here I select one the entries with number of counts greater than a certain number, both for left ansd right time limit

    timeindex = np.sum(counts,1)

    idxa = (np.where((timeindex>200)))
    idxb = (np.where((timeindex>200)))
    a = time[idxa] 
    b = time[idxb]

    t1 = a[0] 
    t2 = b[-1] 
    #t1 = 60.29 ##for disruption shot
    #t2 = 65


#Here I define the length and the area of collimators

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
    #Ha[:,9]=0
    #Va[:,8]=0
    #ind=10
 
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
    
    Hfake = np.multiply(H,areas[0:10]**2) /(Len[0:10]**2)
    Vfake = np.multiply(V,areas[10:19]**2) / (Len[10:19]**2)  
        
#    plt.figure(1)
#    plt.plot(Hchannel,np.sum(Hfake,0),'red')
#    plt.plot(Vchannel,np.sum(Vfake,0),'blue')
#    plt.xlabel('LOS channel')
#    plt.ylabel('Arb')
#    plotc = np.linspace(1,19,19)
#    plt.xticks(plotc)
#    plt.title(str(shot[w]))   
#    plt.savefig('test_' + str(shot[w]) + '.png')


    momentH = np.zeros(len(t))
    momentV = np.zeros(len(t))
    numbercounts = np.zeros(len(t))
    all_zeros = np.zeros(len(t))

#IF THERE ARE ARRAYS WITH ALL ZERO ELEMENTS I FILL THEM WITH THE PREVIOUS NUMBER OF COUNTS

    for i in range(0, len(momentH)):
        all_zeros[i] = not np.any(H[i,:])
        if all_zeros[i] == True:
            #H[i,:] = 100*np.ones((len(Hchannel)))
            H[i,:] = H[i-1,:]

    for i in range(0, len(momentV)):
        all_zeros[i] = not np.any(V[i,:])
        if all_zeros[i] == True:
            #V[i,:] = 100*np.ones((len(Vchannel)))
            V[i,:] = V[i-1,:]
        
# HERE I CALCULATE THE FIRST MOMENT        

    for i in range(0, len(momentH)):
        momentH[i] = np.average(Hchannel, weights = H[i,:])
        momentV[i] = np.average(Vchannel, weights = V[i,:])
        numbercounts[i]  = np.sum(H[i,:])  

# THE OBTAINED SIGNAL IS CRAP SO I SMOOTH IT WITH A SAVITZKY-GOLAY FILTER

    filtr=signal.savgol_filter(momentV, filt, 3)
    filtz=signal.savgol_filter(momentH, filt, 3)

    a1 = np.zeros((len(t)))
    a2 = np.zeros((len(t)))
    a4 = np.zeros((len(t)))
    a5 = np.zeros((len(t)))
    
#this is calculated according to https://iopscience.iop.org/article/10.1088/1748-0221/14/09/C09001/pdf

    for i in range(0, len(t)):
        a1[i] = H[i,0] + H[i,1] + H[i,2] + H[i,3] 
        a2[i] = H[i,4] + H[i,5] + H[i,6] + H[i,7] + H[i,8] + H[i,9]
        a4[i] = V[i,0] + V[i,1] + V[i,2] + V[i,3] 
        a5[i] = V[i,4] + V[i,5] + V[i,6] + V[i,7] + V[i,8] 
   
    asimh = (a1 - a2) / (a1 + a2)
    asimv = (a5 - a4) / (a4 + a5)    

    errasimh = (2*np.sqrt(a1*a2))/(np.sqrt((a1+a2)**3))
    errasimv = (2*np.sqrt(a5*a4))/(np.sqrt((a4+a5)**3))
    
# AGAIN FILTERING 
    asimhf=signal.savgol_filter(asimh, filt, 3)
    asimvf=signal.savgol_filter(asimv, filt, 3)

    
    refit_new = np.interp(t, time_mag, refit)
    zefit_new = np.interp(t, time_mag, zefit)
    
    zefit_big = np.append(zefit_big, zefit_new)
    refit_big = np.append(refit_big, refit_new)
    momentH_big = np.append(momentH_big, momentH)
    momentV_big = np.append(momentV_big, momentV)
    asimhf_big = np.append(asimhf_big, asimhf)
    asimvf_big = np.append(asimvf_big, asimvf)
    errasimh_big = np.append(errasimh_big, errasimh)
    errasimv_big = np.append(errasimv_big, errasimv)
    
    
    ### PLOT THAT SHOWS THE ABSOLUTE COMPARISON AS A FUNCTION OF TIME  #######
    
#    plt.figure(100,figsize = (17, 10))
#    plt.subplot(1,2,1)
#    plt.plot(t, (asimhf-fitaH.intercept_)/fitaH.coef_[0], 'b', linewidth=2,label = 'Asym H', zorder=10)
#    plt.plot(t, (asimhf-(overaHfit))/overbHfit, 'g', linewidth=2,label = 'Asym H', zorder=15)
#    plt.plot(time_mag, zefit,'k', linewidth=2,label = 'Z EFIT')  
#    plt.plot(t, (asimhf-(overaHfit-errorcoeff(acoeffHerr)))/(overbHfit+errorcoeff(bcoeffHerr)), 'g--', linewidth=2,label = 'lower err', zorder=5)
#    plt.plot(t, (asimhf-(overaHfit+errorcoeff(acoeffHerr)))/(overbHfit-errorcoeff(bcoeffHerr)), 'g--', linewidth=2,label = 'upper err', zorder=5)      
#    plt.xlabel('time (s)', fontsize=20)
#    plt.ylabel('Z (m)', fontsize=20)
#    plt.legend(['Asymmetry H','Asymmetry H overall','Z EFIT'])
#    plt.xlim(t1, t2)
#    plt.ylim(0.18, 0.36)
#    plt.xticks(fontsize=14)
#    plt.yticks(fontsize=14)
#    
#    plt.subplot(1,2,2)
#    plt.plot(t, (asimvf-fitaV.intercept_)/fitaV.coef_[0], 'r', linewidth=2,label = 'Asym V', zorder=10)
#    plt.plot(t, (asimvf-(overaVfit))/overbVfit, 'g', linewidth=2,label = 'Asym H', zorder=15)
#    plt.plot(time_mag, refit,'k', linewidth=2,label = 'R EFIT')
#    plt.plot(t, (asimvf-(overaVfit-errorcoeff(acoeffVerr)))/(overbVfit+errorcoeff(bcoeffVerr)), 'g--', linewidth=2,label = 'lower err', zorder=5)
#    plt.plot(t, (asimvf-(overaVfit+errorcoeff(acoeffVerr)))/(overbVfit-errorcoeff(bcoeffVerr)), 'g--', linewidth=2,label = 'upper err', zorder=5)
#    plt.plot(time_mag, refit,'k', linewidth=2,label = 'R EFIT')
#    plt.xlabel('time (s)', fontsize=20)
#    plt.ylabel('R (m)', fontsize=20)
#    plt.legend(['Asymmetry V','Asymmetry V overall','R EFIT'])
#    plt.xlim(t1, t2)
#    plt.ylim(2.85, 3.25)
#    plt.xticks(fontsize=14)
#    plt.yticks(fontsize=14)    
#    
#    plt.savefig('test' + str(shot[w]) + '_' + str(filt) + '_abs.png')#    
    
    # HERE I AM PLOTTING THE ASYMMETRY WITH THE NBI TRACES IN ORDER TO CHECK WEIRD BEHAVIOUR DUE TO MISSING NBI POWER
    
    plotH = Hfake.transpose().copy()
    plotH[np.where(plotH<10)] = np.nan
    
    plotV = Vfake.transpose().copy()
    plotV[np.where(plotV<10)] = np.nan 
        
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
    plt.plot(nbi_time,nbi, 'k')
    plt.ylabel('NBI  (MW)', fontsize=20)
    plt.xlim(t1, t2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.yticks(fontsize=16)
    
    ax2=plt.subplot(gs[2], sharex = ax0)   
    plt.plot(t, (asimhf-fitaH.intercept_)/fitaH.coef_[0], 'b', linewidth=2,label = 'Asym H', zorder=10)
    plt.plot(t, (asimhf-(overaHfit))/overbHfit, 'g', linewidth=2,label = 'Asym H', zorder=15)
    plt.plot(time_mag, zefit,'k', linewidth=2,label = 'Z EFIT')  
    plt.plot(t, (asimhf-(overaHfit-np.std(acoeffH)))/(overbHfit+np.std(bcoeffH)), 'g--', linewidth=2,label = 'lower err', zorder=5)
    plt.plot(t, (asimhf-(overaHfit+np.std(acoeffH)))/(overbHfit-np.std(bcoeffH)), 'g--', linewidth=2,label = 'upper err', zorder=5)      
    #plt.plot(t, (asimhf-(overaHfit-sigma(acoeffH,overaHfit)))/(overbHfit+sigma(bcoeffH,overbHfit)), 'g--', linewidth=2,label = 'lower err', zorder=5)
    #plt.plot(t, (asimhf-(overaHfit+sigma(acoeffH,overaHfit)))/(overbHfit-sigma(bcoeffH,overbHfit)), 'g--', linewidth=2,label = 'upper err', zorder=5) 
    #plt.plot(t, (asimhf-fitaH.intercept_+erraH)/fitaH.coef_[0], 'b--', linewidth=2)
    #plt.plot(t, (asimhf-fitaH.intercept_-erraH)/fitaH.coef_[0], 'b--', linewidth=2)
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('Z (m)', fontsize=20)
    plt.legend(['Asymmetry H','Asymmetry H overall','Z EFIT'],fontsize=16,frameon=False)
    plt.xticks(np.linspace(t1,t2,6),fontsize=18)
    plt.yticks(np.linspace(0.19, 0.35,9), fontsize=18)
    plt.xlim(t1, t2)
    plt.ylim(0.18, 0.36)   
    plt.setp(ax2.get_xticklabels(), visible=False)

    
    plt.subplot(gs[3], sharex = ax0)   
    plt.plot(t, 100*(((asimhf-fitaH.intercept_)/fitaH.coef_[0])-(zefit_new)), 'r', linewidth=2,label = 'Asym V', zorder=10)    
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('diff (cm)', fontsize=20)
    plt.xticks(np.linspace(t1,t2,6),fontsize=16)
    plt.yticks(np.linspace(-10,10,7), fontsize=16)
    plt.xlim(t1, t2)      
    plt.subplots_adjust(hspace=.0)
    plt.ylim(-10, 10)
    plt.subplots_adjust(hspace=.0)
    
    #plt.savefig('zmissingNBI_'+ str(ind) + 'and' + str(ind+10) + '_' + str(shot[w]) + '_' + str(filt) + '_H.png')
    plt.savefig('zNBI_'+ str(shot[w]) + '_' + str(filt) + '_H.png')

    
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
    plt.ylabel('NBI  (MW)', fontsize=20)
    plt.xlim(t1, t2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.yticks(fontsize=16)
    
    ax2=plt.subplot(gs[2], sharex = ax0)   
    plt.plot(t, (asimvf-fitaV.intercept_)/fitaV.coef_[0], 'r', linewidth=2,label = 'Asym V', zorder=10)
    plt.plot(t, (asimvf-(overaVfit))/overbVfit, 'g', linewidth=2,label = 'Asym V', zorder=15)
    plt.plot(time_mag, refit,'k', linewidth=2,label = 'R EFIT')
    plt.plot(t, (asimvf-(overaVfit-np.std(acoeffV)))/(overbVfit+np.std(bcoeffV)), 'g--', linewidth=2,label = 'lower err', zorder=5)
    plt.plot(t, (asimvf-(overaVfit+np.std(acoeffV)))/(overbVfit-np.std(bcoeffV)), 'g--', linewidth=2,label = 'upper err', zorder=5)
    #plt.plot(t, (asimvf-(overaVfit-sigma(acoeffV,overaVfit)))/(overbVfit+sigma(bcoeffV,overbVfit)), 'g--', linewidth=2,label = 'lower err', zorder=5)
    #plt.plot(t, (asimvf-(overaVfit+sigma(acoeffV,overaVfit)))/(overbVfit-sigma(bcoeffV,overbVfit)), 'g--', linewidth=2,label = 'upper err', zorder=5)
    #plt.plot(t, (asimvf-fitaV.intercept_+erraV)/fitaV.coef_[0], 'r--', linewidth=2)
    #plt.plot(t, (asimvf-fitaV.intercept_-erraV)/fitaV.coef_[0], 'r--', linewidth=2)
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('R (m)', fontsize=20)
    plt.legend(['Asymmetry V','Asymmetry V overall','R EFIT'],fontsize=16,frameon=False)
    plt.xticks(np.linspace(t1,t2,6),fontsize=16)
    plt.yticks(np.linspace(2.90,3.20,7), fontsize=16)
    plt.xlim(t1, t2)
    plt.ylim(2.85, 3.25)        
    plt.subplots_adjust(hspace=.0)
    plt.setp(ax2.get_xticklabels(), visible=False)

    
    plt.subplot(gs[3], sharex = ax0)   
    plt.plot(t, 100*(((asimvf-fitaV.intercept_)/fitaV.coef_[0])-(refit_new)), 'r', linewidth=2,label = 'Asym V', zorder=10)
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('diff (cm)', fontsize=20)
    plt.xticks(np.linspace(t1,t2,6),fontsize=16)
    plt.yticks(np.linspace(-10,10,7), fontsize=16)
    plt.xlim(t1, t2)   
    plt.subplots_adjust(hspace=.0)
    plt.ylim(-10, 10)
    
    
    #plt.savefig('zmissingNBI_'+ str(ind) + 'and' + str(ind+10) + '_' + str(shot[w]) + '_' + str(filt) + '_V.png')
    plt.savefig('zNBI_'+ str(shot[w]) + '_' + str(filt) + '_V.png')

    
# HERE I DO THE LINEAR REGRESSION TO ALL POINTS 
    ##### THOSE ARE FOR THE MOMENTS #######
fitmH = LinearRegression().fit(zefit_big.reshape((-1, 1)), momentH_big)
fitmV = LinearRegression().fit(refit_big.reshape((-1, 1)), momentV_big)

    ##### THOSE ARE FOR THE ASYMMETRY #######
fitaH = LinearRegression().fit(zefit_big.reshape((-1, 1)), asimhf_big, sample_weight = 1/errasimh_big**2)
fitaV = LinearRegression().fit(refit_big.reshape((-1, 1)), asimvf_big, sample_weight = 1/errasimv_big**2)

r_sqmH = fitmH.score(zefit_big.reshape((-1, 1)), momentH_big)
r_sqmV = fitmV.score(refit_big.reshape((-1, 1)), momentV_big)
r_sqaH = fitaH.score(zefit_big.reshape((-1, 1)), asimhf_big)
r_sqaV = fitaV.score(refit_big.reshape((-1, 1)), asimvf_big)

Z = np.linspace(np.min(zefit_big), np.max(zefit_big), 1000)
R = np.linspace(np.min(refit_big), np.max(refit_big), 1000)

##### PLOT THAT SHOWS THE OVERALL LINEAR FIT  #######

plt.figure(400,figsize = (24, 17))
#plt.suptitle(shot, fontsize=26)

plt.subplot(2,2,1)
plt.text(np.max(zefit_big), np.max(momentH_big), '$R^2$ = (%.3f)'%r_sqmH,
         verticalalignment='top', horizontalalignment='right', fontsize=25) 
plt.scatter(zefit_big, momentH_big, c='b', s=10)
plt.plot(Z, fitmH.intercept_+Z*fitmH.coef_, c='r', linewidth=3)
plt.xlabel('Z (m)', fontsize=20)
plt.ylabel('Moment H', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
   
plt.subplot(2,2,2)
plt.text(np.max(refit_big), np.max(momentV_big), '$R^2$ = (%.3f)'%r_sqmV,
         verticalalignment='top', horizontalalignment='right', fontsize=25) 
plt.scatter(refit_big, momentV_big, c='r', s=10)
plt.plot(R, fitmV.intercept_+R*fitmV.coef_, c='b', linewidth=3)
plt.xlabel('R (m)', fontsize=20)
plt.ylabel('Moment V', fontsize=20)    
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
    
plt.subplot(2,2,3)
plt.plot(Z, fitaH.intercept_+Z*fitaH.coef_, c='r', linewidth=3, zorder=10)
plt.plot(Z, (fitaH.intercept_+erraH) + Z*fitaH.coef_, 'r--', linewidth=2)
plt.plot(Z, (fitaH.intercept_-erraH) + Z*fitaH.coef_, 'r--', linewidth=2)
plt.text(np.max(zefit_big), np.min(asimhf_big), '$R^2$ = (%.3f)'%r_sqaH,
         verticalalignment='bottom', horizontalalignment='right', fontsize=25) 
plt.scatter(zefit_big, asimhf_big, c='b', s=10)
plt.errorbar(zefit_big, asimhf_big, yerr = errasimh_big, fmt='o', c='k', zorder=0)
plt.xlabel('Z (m)', fontsize=20)
plt.ylabel('Asymmetry H', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
    
plt.subplot(2,2,4)
plt.text(np.max(refit_big), np.min(asimvf_big), '$R^2$ = (%.3f)'%r_sqaV,
         verticalalignment='bottom', horizontalalignment='right', fontsize=25) 
plt.scatter(refit_big, asimvf_big, c='r', s=10)
plt.errorbar(refit_big, asimvf_big, yerr = errasimv_big, fmt='o', c='k', zorder=0)
plt.plot(R, fitaV.intercept_+ R*fitaV.coef_, c='b', linewidth=3, zorder=10)
plt.plot(R, (fitaV.intercept_+erraV)+ R*fitaV.coef_, 'b--', linewidth=2)
plt.plot(R, (fitaV.intercept_-erraV)+ R*fitaV.coef_, 'b--', linewidth=2)
plt.xlabel('R (m)', fontsize=20)
plt.ylabel('Asymmetry V', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


#plt.savefig('zmissing_' + str(ind) + 'and' + str(ind+10) + '_' + str(shot) + '_' + str(filt) + '.png')

plt.close('all')

errbH = errorFitb(asimhf_big, zefit_big, fitaH.intercept_ , fitaH.coef_[0])
errbV = errorFitb(asimvf_big, refit_big, fitaV.intercept_ , fitaV.coef_[0])

erraH = errorFita(asimhf_big, zefit_big, fitaH.intercept_ , fitaH.coef_[0])
erraV = errorFita(asimvf_big, refit_big, fitaV.intercept_ , fitaV.coef_[0])


print('The a coefficient for the horizontal camera is = ', fitaH.intercept_, '+/-' , erraH)
print('The b coefficient for the horizontal camera is = ',  fitaH.coef_[0], '+/-' , errbH)

print('The a coefficient for the vertical camera is = ',  fitaV.intercept_, '+/-' , erraV)
print('The b coefficient for the vertical camera is = ',    fitaV.coef_[0], '+/-' , errbV) 


acoeffH  =   np.array((-0.573669,-0.57143,-0.64288,-0.600755,-0.514923,-0.579144,-0.4264))
acoeffHerr = np.array((0.005, 0.006, 0.0189, 0.004, 0.0049, 0.0035, 0.00109))
bcoeffH    = np.array((2.312,2.2823,2.5040,2.3162,2.0107,2.5218, 1.722))
bcoeffHerr = np.array((0.038,0.0228, 0.01369,0.03,0.037,0.0258,0.08))

acoeffV    = np.array((-4.88075,-4.722103,-2.7719349,-4.58529, -3.87109,-2.967226, -3.28365))
acoeffVerr = np.array((0.03186,0.06,0.03,0.0573,0.073,0.02325,0.0394))
bcoeffV    = np.array((1.63554,1.5782,0.9309,1.52711,1.29485,1.0037,1.1059))
bcoeffVerr = np.array((0.02,0.02,0.02,0.037,0.048,0.015,0.026))

overaHfit = np.mean(acoeffH)
overbHfit = np.mean(bcoeffH)
overaVfit = np.mean(acoeffV)
overbVfit = np.mean(bcoeffV)



#overaHfit = np.average(acoeffH, weights = (1/acoeffHerr**2))
#overbHfit = np.average(bcoeffH, weights = (1/bcoeffHerr**2))
#overaVfit = np.average(acoeffV, weights = (1/acoeffVerr**2))
#overbVfit = np.average(bcoeffV, weights = (1/bcoeffVerr**2))




#    
#    plt.figure(1)
#    plt.plot(Hchannel,np.sum(Hfake,0),'red')
#    plt.plot(Vchannel,np.sum(Vfake,0),'blue')
#    plt.xlabel('LOS channel')
#    plt.ylabel('Arb')
#    plotc = np.linspace(1,10,10)
#    plt.xticks(plotc)
#    plt.title(str(shot))
#    plt.show()    
    
#    fig = plt.figure(100, figsize = (17, 17))
#    fig.suptitle(str(shot[w]), fontsize=26)
#
#    ax=plt.subplot(2,2,1)
#    ax.set_xlabel('time (s)', fontsize=14)
#    ln1 = ax.plot(time_mag, zefit,'k', linewidth=2,label = 'Z EFIT')
#    ax.set_ylabel('Z (m)', fontsize=14)
#    ax2=ax.twinx()
#    ln2 = ax2.plot(t, filtz,'r', linewidth=2,label = 'NCH')
#    ax2.set_ylabel('First moment', fontsize=14)
## added these three lines
#    lns = ln1+ln2
#    labs = [l.get_label() for l in lns]
#    ax.legend(lns, labs, loc=0, frameon=False)
#
##
#    ax=plt.subplot(2,2,2)
#    ax.set_xlabel('time (s)', fontsize=14)
#    ln1 = ax.plot(time_mag, refit,'k', linewidth=2,label = 'R EFIT')
#    ax.set_ylabel('R (m)', fontsize=14)
#    ax2=ax.twinx()
#    ln2 = ax2.plot(t, filtr,'r', linewidth=2,label = 'NCV')
#    ax2.set_ylabel('First moment', fontsize=14)
## added these three lines
#    lns = ln1+ln2
#    labs = [l.get_label() for l in lns]
#    ax.legend(lns, labs, loc=0, frameon=False)
#
#
#    ax = plt.subplot(2,2,3)
#    ax.set_xlabel('time (s)', fontsize=14)
#    ln1 = ax.plot(time_mag, zefit,'k', linewidth=2,label = 'Z EFIT')
#    ax.set_ylabel('Z (m)', fontsize=14)
#    ax2=ax.twinx()
#    ln2 = ax2.plot(t, asimhf, 'r', linewidth=2,label = 'Asym')
#    ax2.set_ylabel('Asymmetry', fontsize=14)
## added these three lines
#    lns = ln1+ln2
#    labs = [l.get_label() for l in lns]
#    ax.legend(lns, labs, loc=0, frameon=False)
#
#
#    ax = plt.subplot(2,2,4)
#    ax.set_xlabel('time (s)', fontsize=14)
#    ln1 = ax.plot(time_mag, refit,'k', linewidth=2,label = 'R EFIT')
#    ax.set_ylabel('R (m)', fontsize=14)
#    ax2=ax.twinx()
#    ln2 = ax2.plot(t, asimvf, 'r', linewidth=2,label = 'Asym')
#    ax2.set_ylabel('Asymmetry', fontsize=14)
## added these three lines
#    lns = ln1+ln2
#    labs = [l.get_label() for l in lns]
#    ax.legend(lns, labs, loc=0, frameon=False)
#
#    plt.savefig('test1.png')
#    plt.savefig(str(shot[w]) + '.png')