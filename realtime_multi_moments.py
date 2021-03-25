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

#error on the linear regression coefficients ####
def errorFitb(asi, A, efit, B):
    return ((1/(len(asi)-2))* np.sum((asi - A - efit*B)**2)  / np.sum((efit-np.mean(efit))**2))**0.5   

def errorFita(errb, asi, efit):
    return errb*((1/len(asi)) * np.sum(efit**2))**0.5 



##### POISSON ERROR ON NC COUNTS ####
def errcounts(l,area,los):
    return ((l**2)/(area**2))*(np.abs(los)**0.5)

def errorcoeff(we):
    return np.sqrt(1/np.sum(1/we**2))


def sigma(fact, avgfact):
    return np.sqrt((1/len(fact)) * np.sum((fact-avgfact)**2)     )

zefit_big = []
refit_big = []
momentH_big = []
momentV_big = []

#shot = [86658, 86659,86660,86661,86662,86663,86664,86667,86668,86669,86670,86671,86672,86673,86674,86675,86676,86677,86678,86679] 
#filt=205 #countsleft>1000 #countsright>1000

############ THESE ARE GREAT ##########
shot = [84536,84539, 84540, 84541,84542,84543,84544,84545] ; scenario=1
filt=47; countsleft=200 ; countsright=200
#shot = [84786,84788,84789,84792,84793,84794,84796,84797,84798]  ; scenario=2
#filt=45 ;countsleft=200 ;countsright=200
#shot = [84802,84803, 84808] ; scenario=3
#filt= 85 ;countsleft=500 ;countsright=2000
#shot = [85197, 85201, 85203, 85204, 85205]  ; scenario=4
#filt=21 ;countsleft=200 ;countsright=2000
#shot = [85393, 85394, 85398, 85399] ; scenario=5
#filt=15 ;countsleft=200 ;countsright=500
#shot = [92393, 92394, 92395, 92398, 92399] ; scenario=6
#filt = 45 ;countsleft=200 ;countsright=5800
#shot=[94256,94257,94259,94261,94262,94263,94264,94265,94266,94268,94270] ; scenario=7
#filt = 85 ;countsleft=50 ;countsright=50

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

    idxa = (np.where((timeindex>countsleft)))
    idxb = (np.where((timeindex>countsright)))
    a = time[idxa] 
    b = time[idxb]

    t1 = a[0] 
    t2 = b[-1] 
    #t1 = 60.29
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
    
    errH={}
    errV={}

    
    refit_new = np.interp(t, time_mag, refit)
    zefit_new = np.interp(t, time_mag, zefit)
    
    zefit_big = np.append(zefit_big, zefit_new)
    refit_big = np.append(refit_big, refit_new)
    momentH_big = np.append(momentH_big, momentH)
    momentV_big = np.append(momentV_big, momentV)
    

    
    # HERE I AM PLOTTING THE ASYMMETRY WITH THE NBI TRACES IN ORDER TO CHECK WEIRD BEHAVIOUR DUE TO MISSING NBI POWER
    
    plotH = Hfake.transpose().copy()
    plotH[np.where(plotH<10)] = np.nan
    
    plotV = Vfake.transpose().copy()
    plotV[np.where(plotV<10)] = np.nan 
        
    [T,CHH] = np.meshgrid(t, np.linspace(1,10,10)) 

    # HERE I AM PLOTTING THE ASYMMETRY WITH THE NBI TRACES IN ORDER TO CHECK WEIRD BEHAVIOUR DUE TO MISSING NBI POWER
    
    plt.figure(200,figsize = (17, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1,1]) 
    
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
    
    plt.subplot(gs[2], sharex = ax0)   
    plt.plot(t, (filtz-fitmH.intercept_)/fitmH.coef_[0], 'b', linewidth=2,label = 'Moment', zorder=10)
    plt.plot(t, zefit_new,'k', linewidth=2,label = 'Z EFIT')  
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('Z (m)', fontsize=20)
    plt.legend(['Moment Z','Z EFIT'],fontsize=16,frameon=False)
    plt.xticks(np.linspace(t1,t2,6),fontsize=18)
    plt.yticks(np.linspace(0.19, 0.35,9), fontsize=18)
    plt.xlim(t1, t2)
    plt.ylim(0.18, 0.36)   
    
    
    
    plt.subplot(gs[3], sharex = ax0)   
    plt.plot(t, 100*(((filtz-fitmH.intercept_)/fitmH.coef_[0])-(zefit_new)), 'b', linewidth=2,label = 'single asym', zorder=10)    
    #plt.plot(t, 100*(((asimhf-overaHfit)/overbHfit)-(zefit_new)), 'g', linewidth=2,label = 'total asym', zorder=12)
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('diff (cm)', fontsize=20)
    plt.xticks(np.linspace(t1,t2,6),fontsize=16)
    plt.yticks(np.linspace(-10,10,7), fontsize=16)
    plt.xlim(t1, t2)      
    plt.subplots_adjust(hspace=.0)
    plt.ylim(-10, 10)
    plt.subplots_adjust(hspace=.0)

    plt.savefig('zmomentNBI_' + str(shot[w]) + '_' + str(filt) + '_H.png')
    
    
    [T,CHV] = np.meshgrid(t, np.linspace(11,19,9))    

    
    plt.figure(300,figsize = (17, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1,1])  
    
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
    
    plt.subplot(gs[2], sharex = ax0)   
    plt.plot(t, (filtr-fitmV.intercept_)/fitmV.coef_[0], 'b', linewidth=2,label = 'Moment', zorder=10)
    plt.plot(t, refit_new,'k', linewidth=2,label = 'R EFIT')
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('R (m)', fontsize=20)
    plt.legend(['Moment V','R EFIT'],fontsize=16,frameon=False)
    plt.xticks(np.linspace(t1,t2,6),fontsize=16)
    plt.yticks(np.linspace(2.90,3.20,7), fontsize=16)
    plt.xlim(t1, t2)
    plt.ylim(2.85, 3.25)        
    plt.subplots_adjust(hspace=.0)
    
    plt.subplot(gs[3], sharex = ax0)   
    plt.plot(t, 100*(((filtr-fitmV.intercept_)/fitmV.coef_[0])-(refit_new)), 'b', linewidth=2,label = 'single asym', zorder=10)    
    #plt.plot(t, 100*(((asimhf-overaHfit)/overbHfit)-(zefit_new)), 'g', linewidth=2,label = 'total asym', zorder=12)
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('diff (cm)', fontsize=20)
    plt.xticks(np.linspace(t1,t2,6),fontsize=16)
    plt.yticks(np.linspace(-10,10,7), fontsize=16)
    plt.xlim(t1, t2)      
    plt.subplots_adjust(hspace=.0)
    plt.ylim(-10, 10)
    plt.subplots_adjust(hspace=.0)

    plt.savefig('zomentNBI_' + str(shot[w]) + '_' + str(filt) + '_V.png')

    
# HERE I DO THE LINEAR REGRESSION TO ALL POINTS 
    ##### THOSE ARE FOR THE MOMENTS #######
fitmH = LinearRegression().fit(zefit_big.reshape((-1, 1)), momentH_big)
fitmV = LinearRegression().fit(refit_big.reshape((-1, 1)), momentV_big)



r_sqmH = fitmH.score(zefit_big.reshape((-1, 1)), momentH_big)
r_sqmV = fitmV.score(refit_big.reshape((-1, 1)), momentV_big)


Z = np.linspace(np.min(zefit_big), np.max(zefit_big), 1000)
R = np.linspace(np.min(refit_big), np.max(refit_big), 1000)

##### PLOT THAT SHOWS THE OVERALL LINEAR FIT  #######

plt.figure(400,figsize = (24, 17))
#plt.suptitle(shot, fontsize=26)

plt.subplot(1,2,1)
plt.text(np.max(zefit_big), np.max(momentH_big), '$R^2$ = (%.3f)'%r_sqmH,
         verticalalignment='top', horizontalalignment='right', fontsize=25) 
plt.scatter(zefit_big, momentH_big, c='b', s=10)
plt.plot(Z, fitmH.intercept_+Z*fitmH.coef_, c='r', linewidth=3)
plt.xlabel('Z (m)', fontsize=20)
plt.ylabel('Moment H', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
   
plt.subplot(1,2,2)
plt.text(np.max(refit_big), np.max(momentV_big), '$R^2$ = (%.3f)'%r_sqmV,
         verticalalignment='top', horizontalalignment='right', fontsize=25) 
plt.scatter(refit_big, momentV_big, c='r', s=10)
plt.plot(R, fitmV.intercept_+R*fitmV.coef_, c='b', linewidth=3)
plt.xlabel('R (m)', fontsize=20)
plt.ylabel('Moment V', fontsize=20)    
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
    

plt.savefig('zmoment'+ str(shot) + '_' + str(filt) + '.png')

plt.close('all')

errbH = errorFitb(momentH_big, fitmH.intercept_, zefit_big, fitmH.coef_)
errbV = errorFitb(momentV_big, fitmV.intercept_, refit_big, fitmV.coef_)
erraH = errorFita(errbH, momentH_big, zefit_big)
erraV = errorFita(errbV, momentV_big, refit_big)

print('The a coefficient for the horizontal camera is = ', fitmH.intercept_, '+/-' , erraH)
print('The b coefficient for the horizontal camera is = ',  fitmH.coef_[0], '+/-' , errbH)

print('The a coefficient for the vertical camera is = ',  fitmV.intercept_, '+/-' , erraV)
print('The b coefficient for the vertical camera is = ',    fitmV.coef_[0], '+/-' , errbV) 


