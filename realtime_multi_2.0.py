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


acoeffH  =   np.array((-0.579,-0.559,-0.597,-0.542))
acoeffHerr = np.array((0.003, 0.002, 0.002, 0.002))
bcoeffH    = np.array((2.33,2.25,2.302,2.18))
bcoeffHerr = np.array((0.01,0.01, 0.01,0.01))

acoeffV    = np.array((5.29,5.8,4.5, 4.12))
acoeffVerr = np.array((0.02,0.015,0.04,0.03))
bcoeffV    = np.array((-1.697,-1.86,-1.432,-1.31))
bcoeffVerr = np.array((0.01,0.005,0.01,0.01))




acoeffHm  =   np.array((5.95,5.81,5.847,5.72))
acoeffHmerr = np.array((0.015, 0.01, 0.01, 0.01))
bcoeffHm    = np.array((-5.58,-5.1,-5.09,-4.65))
bcoeffHmerr = np.array((0.05,0.03,0.03,0.03))

acoeffVm    = np.array((5.59,4.82, 6.77, 7.68))
acoeffVmerr = np.array((0.08,0.07,0.1,0.08))
bcoeffVm    = np.array((3.03,3.27,2.63,2.34))
bcoeffVmerr = np.array((0.03,0.02,0.04,0.02))

overaHfit = np.mean(acoeffH)
overbHfit = np.mean(bcoeffH)
overaVfit = np.mean(acoeffV)
overbVfit = np.mean(bcoeffV)

overaHfitm = np.mean(acoeffHm)
overbHfitm = np.mean(bcoeffHm)
overaVfitm = np.mean(acoeffVm)
overbVfitm = np.mean(bcoeffVm)

overaHfit = -0.513
overbHfit = 2.057
overaVfit = 2.63
overbVfit = -0.951
#
overaHfitm = 5.444
overbHfitm = -3.68
overaVfitm = 10.81
overbVfitm = 1.3

########### I FEEL LIKE IT IS TOO SOON TO INCLUDE THOSE SINCE FUNNY THINGS ARE GOING ON  ###########
#shot = [86658, 86659,86660,86661,86662,86663,86664,86667,86668,86669,86670,86671,86672,86673,86674,86675,86676,86677,86678,86679] 
#filt=295; countsleft=2000 ;countsright=2000


#shot = [94665,94666,94667,94669,94670,94671,94672,94674,94675,94676,94677,94678,94679,94680,94681,94682,94683,94684] 
#filt=285; countsleft=200 ;countsright=200


######################
#shot = [61167]  # THIS IS THE NICE ONE WITH THE DISPLACEMENT
#filt = 21 ; countsleft=20 ; countsright=20
#shot = [92398] 
#filt = 115 ; countsleft=100 ; countsright=100

#shot = [85186]
#filt = 95 ; countsleft=100 ; countsright=100

#
#shot = [84802, 84803, 84804] ; scenario=3
#filt=485; countsleft=2000; countsright=5000

#shot=[94256,94257,94259,94261,94262,94263,94264,94265,94266,94268,94270] ; scenario=6
#filt = 851 ;countsleft=5 ;countsright=5
#shot = [92393, 92394, 92395, 92398, 92399] ; scenario=5
#filt = 451 ;countsleft=20 ;countsright=580
#shot=[94270]; filt=251 ;countsleft=20 ;countsright=580

#shot = [94665]
#filt=451 ;countsleft=200 ;countsright=200

#shot = [53761]
#filt=5 ;countsleft=200 ;countsright=200

############ THESE ARE GREAT ##########
#shot = [84536,84539, 84540, 84541,84542,84543,84544,84545] ; scenario=1
#filt=471; countsleft=200 ; countsright=200
#shot = [84786,84788,84789,84792,84796,84797,84798]  ; scenario=2
#filt=451 ;countsleft=200 ;countsright=200
#shot = [85197, 85201, 85203, 85204, 85205]  ; scenario=3
#filt=211 ;countsleft=20 ;countsright=200
#shot = [85393, 85394, 85399,85400, 85401, 85402, 85403] ; scenario=4
#filt=151 ;countsleft=20 ;countsright=2000

#shot=[82307,82312,82338,82342,82551,82552,82553,82571,82572,82575,82578,82579,82730,82758,82807,82810,82813,82814,82815,82816,82817,82818,
#      82820,82821,82828,82829,82831,82833,82859,82861,82862,82863,82866,82870,82870,82872,82874,82876,82877,82878,82878,82880,82883,82894,
#      82895,82896,82937,83014,83042,83043,83044,83045,83180,83181,83182,83183,83187,83188,83618, 83631,83633,84495,84496,84497,84498,84499,
#      84500,84501,84502,84503, 84504,84506,84507,84508,84509,84510,84511,84512,84513, 84514, 84515, 84516, 84517, 84519,
#      84536,84538,84539, 84540, 84541,84542,84543,84544,84545,84786,84787,84788,84789,84792,84793,84794,84795,84796,84797,84798,
#      84801,84802,84803, 84804,84806,84808,85100,85185,85186,85187,85192,85197, 85200, 85201, 85203, 85204, 85205,85208,85209,85300,
#      85302,85303, 85305,85306,85307,85308,85309,85310,85311,85312,85313,85314,85315,85316,85317,85318,85319,
#      85393, 85394, 85398, 85399,85400, 85401,85402,85403, 85406, 85407, 85408, 85409, 86372,86456,86459,86461, 86464, 86614, 86658,
#      86659, 86660, 86661,86662,86663,86664,86667,86668,86669, 86670, 86671, 86672, 86673, 86674, 86675,  86676, 86677, 86678, 86679, 86682,
#      86683, 86684, 86685, 86686, 86687, 86688, 86774,86775,86825, 86826,86888, 88305, 88307, 88309, 88310, 88311,88312, 88312, 88315, 89301, 
#      89302, 89303, 89305, 89306, 89307, 89308, 89309,89310, 89312, 89313, 89314, 89317, 89318, 89319, 90316, 90318, 90319, 90363, 90386,
#      91301, 92205, 92207, 92211,92213,92214, 92351, 92380, 92391, 92393, 92394, 92395, 92396, 92398, 92399,92411, 92412, 92413, 92414, 92431,
#      94209, 94217, 94243, 94249, 94250, 94251, 94252, 94253, 94256,94257,94259,94261,94262,94263,94264,94265,94266,94267, 94268,94270,
#      94271, 94272, 94310, 94323, 94417, 94437, 94441, 94606, 94607, 94610, 94611, 94635,94665, 94666, 94667, 94669, 94670, 94671, 94672, 94674,
#      94675, 94676, 94677, 94678, 94679, 94680, 94681, 94682, 94683, 94684, 94701, 95335, 95649, 95650, 95652, 95655, 95665, 95666, 95667, 95668, 
#      95669, 95671, 95673, 95674, 95675, 95677, 95679, 95680, 95682, 95683, 95684, 95686, 95688, 95689, 95690, 95691, 95692, 95693, 95694, 95697,
#      95698,95699,96058, 96059]
#shot=[53761];countsleft=200 ;countsright=200; filt=285
shot = [94270];countsleft=200 ;countsright=200; filt=285
############ THOSE ARE GREAT ##########

###M18##
#shot = [94416,94417,96730]
#filt=301; countsleft=200 ; countsright=400

#shot = [94267,
#94268,
#94270,
#94665,
#95335,
#95649,
#95652,
#95674,
#96058,
#96059]
#shot=[94701,95335,95649,95650,95652,95655,95665,95666,95667,95668,95669,95671,95673,95674,95675,
 #     95677,95679,95680,95682,95683,95684,95686,95688,95689,95690,95691,95692,95693,95694,95697,
  #    95698,95699, 96058,96059]
#shot=[94665]
#filt = 761 ;countsleft=200 ;countsright=200





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
    
   
    newtime = np.linspace(round(float((time[0]))), round(float((time[-1]))), (round(float((time[-1])))-round(float((time[0]))))*1000)

        

    counts_old = np.reshape(f,(len(time),20))   #NEW NEUTRON CAMERA
    counts = np.zeros((len(newtime),20))
    
    if round(float(np.diff(time)[0]*1000)) > 1.5:
        for i in range(len(counts_old[1])):
            counts[:,i] = round(float(np.diff(time)[0]*1000))*(interpolation(newtime,time,counts_old[:,i])/1000)
        time=newtime
    else:
        counts = np.zeros((len(time),20))
        counts = counts_old
        print('The time resolution is 1 ms, please go ahead with the analysis.')    

# Here I select one the entries with number of counts greater than a certain number, both for left ansd right time limit

    timeindex = np.sum(counts,1)

    idxa = (np.where((timeindex>countsleft)))
    idxb = (np.where((timeindex>countsright)))
    idxnbi = np.where((nbi>1e6))

    a = time[idxa] 
    b = time[idxb]

    if a[0]>nbi_time[idxnbi][0]:
        t1 = a[0] 
    else:
        t1 = nbi_time[idxnbi][0]
        
    if b[-1]<nbi_time[idxnbi][-1]:
        t2 = b[-1] 
    else:            
        t2 = nbi_time[idxnbi][-1]
   
    t1=a[0]
    t2=b[-1]
    
    t1=48.35
    t2=53.07
    
    

    

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
    


# 
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
    
    Hfake = np.multiply(H,areas[0:10]**2) /   (Len[0:10]**2)
    Vfake = np.multiply(V,areas[10:19]**2) / (Len[10:19]**2)  
    H=Hfake
    V=Vfake
    plt.plot(Hchannel,np.sum(H,0),'red')
    plt.plot(Vchannel,np.sum(V,0),'blue')
    plt.scatter(Hchannel,np.sum(H,0),c="k")
    plt.scatter(Vchannel,np.sum(V,0),c="k")
    plt.xlabel('LOS channel')
    plt.ylabel('Arb')
    plotc = np.linspace(1,19,19)
    plt.xticks(plotc)
    plt.title(str(shot[w]))   
    plt.savefig('testc_' + str(shot[w]) + '.png')



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
    plt.plot(t, (asimhf-fitaH.intercept_)/fitaH.coef_[0], 'r', linewidth=2,label = 'Asym H', zorder=10)
    #plt.plot(t, (asimhf-(overaHfit))/overbHfit, 'r', linewidth=2,label = 'Asym H', zorder=15)
    plt.plot(t, (filtz-fitmH.intercept_)/fitmH.coef_[0], 'b', linewidth=2,label = 'Moment', zorder=10)
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
    plt.plot(t, 100*(((asimhf-fitaH.intercept_)/fitaH.coef_[0])-(zefit_new)), 'r', linewidth=2,label = 'asym', zorder=15)  
    #plt.plot(t, 100*(((asimhf-overaHfit)/overbHfit)-(zefit_new)), 'r--', linewidth=2,label = 'total asym', zorder=14)
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
    plt.savefig('zeNBI_'+ str(shot[w]) + '_' + str(filt) + '_H.png')

    
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
    #plt.plot(t, (asimvf-overaVfit)/overbVfit, 'r', linewidth=2,label = 'Asym V', zorder=15)
    #plt.plot(t, (filtr-fitmV.intercept_)/fitmV.coef_[0], 'b', linewidth=2,label = 'Moment', zorder=10)
    plt.plot(t, (filtr-overaVfitm)/overbVfitm, 'b', linewidth=2,label = 'Moment', zorder=10)
    

    
    plt.plot(t, refit_new,'k', linewidth=2,label = 'R EFIT')
    #plt.plot(t, (asimvf-(overaVfit-0.4))/(overbVfit+0.1), 'g--', linewidth=2,label = 'lower err', zorder=5)
    #plt.plot(t, (asimvf-(overaVfit+0.4))/(overbVfit-0.1), 'g--', linewidth=2,label = 'upper err', zorder=5)
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
    plt.plot(t, 100*(((asimvf-fitaV.intercept_)/fitaV.coef_[0])-(refit_new)), 'r', linewidth=2,label = 'asym', zorder=15)  
    #plt.plot(t, 100*(((asimvf-overaVfit)/overbVfit)-(refit_new)), 'r--', linewidth=2,label = 'total asym', zorder=14)
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
    plt.savefig('zeNBI_'+ str(shot[w]) + '_' + str(filt) + '_V.png')
    
    
    x1 = zefit_new[:,np.newaxis]
    x2 = refit_new[:,np.newaxis]
    y1 = (asimhf-overaHfit)/overbHfit
    y2 = (asimvf-overaVfit)/overbVfit
    a1, _, _, _ = np.linalg.lstsq(x1, y1)
    a2, _, _, _ = np.linalg.lstsq(x2, y2)
    #print('H =', a1)
    print('V =', a2)
    
#    np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/' + str(shot[w]) + '_REFIT.txt', refit_new)
#    np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/' + str(shot[w]) + '_ZEFIT.txt', zefit_new)
#    np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/' + str(shot[w]) + '_TIME.txt', t)
#    np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/' + str(shot[w]) + '_ASYH.txt', (asimhf-fitaH.intercept_)/fitaH.coef_[0])
#    np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/' + str(shot[w]) + '_ASYV.txt', (asimvf-fitaV.intercept_)/fitaV.coef_[0])
#    np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/' + str(shot[w]) + '_MOMH.txt', (filtz-fitmH.intercept_)/fitmH.coef_[0])
#    np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/' + str(shot[w]) + '_MOMV.txt', (filtr-fitmV.intercept_)/fitmV.coef_[0])   
#    np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/' + str(shot[w]) + '_ASYOVERH.txt', (asimhf-(overaHfit))/overbHfit)
#    np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/' + str(shot[w]) + '_ASYOVERV.txt', (asimhf-(overaHfit))/overbHfit)
#    np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/' + str(shot[w]) + '_MOMOVERH.txt', (filtz-(overaHfitm))/overbHfitm)
#    np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/' + str(shot[w]) + '_MOMOVERV.txt', (filtr-(overaHfitm))/overbHfitm)
# HERE I DO THE LINEAR REGRESSION TO ALL POINTS 
    ##### THOSE ARE FOR THE MOMENTS #######


    ##### THOSE ARE FOR THE ASYMMETRY #######
indexh = (np.where((errasimh_big==0)))
indexv = (np.where((errasimv_big==0)))

errasimh_big[indexh] = np.mean(errasimh_big)
errasimv_big[indexv] = np.mean(errasimv_big)

    
fitaH = LinearRegression().fit(zefit_big.reshape((-1, 1)), asimhf_big)#, sample_weight = 1/errasimh_big**2)
fitaV = LinearRegression().fit(refit_big.reshape((-1, 1)), asimvf_big)#, sample_weight = 1/errasimv_big**2)

fitmH = LinearRegression().fit(zefit_big.reshape((-1, 1)), momentH_big)#, sample_weight = 1/errmomh_big**2)
fitmV = LinearRegression().fit(refit_big.reshape((-1, 1)), momentV_big)#, sample_weight = 1/errmomv_big**2)

r_sqmH = fitmH.score(zefit_big.reshape((-1, 1)), momentH_big)
r_sqmV = fitmV.score(refit_big.reshape((-1, 1)), momentV_big)
r_sqaH = fitaH.score(zefit_big.reshape((-1, 1)), asimhf_big)
r_sqaV = fitaV.score(refit_big.reshape((-1, 1)), asimvf_big)

Z = np.linspace(np.min(zefit_big), np.max(zefit_big), 1000)
R = np.linspace(np.min(refit_big), np.max(refit_big), 1000)

##### PLOT THAT SHOWS THE OVERALL LINEAR FIT  #######
#
#np.savetxt('S' + str(scenario)+ '_ZEFIT.txt',zefit_big)
#np.savetxt('S' + str(scenario)+ '_REFIT.txt',refit_big)
#np.savetxt('S' + str(scenario)+ '_HASIM.txt',asimhf_big)
#np.savetxt('S' + str(scenario)+ '_VASIM.txt',asimvf_big)


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
#plt.errorbar(zefit_big, asimhf_big, yerr = errasimh_big, fmt='o', c='k', zorder=0)
plt.xlabel('Z (m)', fontsize=20)
plt.ylabel('Asymmetry H', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
    
plt.subplot(2,2,4)
plt.text(np.max(refit_big), np.max(asimvf_big), '$R^2$ = (%.3f)'%r_sqaV,
         verticalalignment='top', horizontalalignment='right', fontsize=25) 
plt.scatter(refit_big, asimvf_big, c='r', s=10)
#plt.errorbar(refit_big, asimvf_big, yerr = errasimv_big, fmt='o', c='k', zorder=0)
plt.plot(R, fitaV.intercept_        + R*fitaV.coef_, c='b', linewidth=3, zorder=10)
plt.plot(R, (fitaV.intercept_+erraV)+ R*fitaV.coef_, 'b--', linewidth=2)
plt.plot(R, (fitaV.intercept_-erraV)+ R*fitaV.coef_, 'b--', linewidth=2)
plt.xlabel('R (m)', fontsize=20)
plt.ylabel('Asymmetry V', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


#np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/S' + str(scenario) + '_REFITM.txt', refit_big)
#np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/S' + str(scenario) + '_ZEFITM.txt', zefit_big)
#np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/S' + str(scenario) + '_ASYV.txt', asimvf_big)
#np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/S' + str(scenario) + '_ASYH.txt', asimhf_big)
#np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/S' + str(scenario) + '_MOMV.txt', momentV_big)
#np.savetxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ALLINFO/S' + str(scenario) + '_MOMH.txt', momentH_big)

plt.savefig('zzzzzzzzza' + str(shot) + '_' + str(filt) + '.png')

plt.close('all')

erraH = errorFita(asimhf_big, zefit_big, fitaH.intercept_ , fitaH.coef_[0])
errbH = errorFitb(asimhf_big, zefit_big, fitaH.intercept_ , fitaH.coef_[0])
erraV = errorFita(asimvf_big, refit_big, fitaV.intercept_ , fitaV.coef_[0])
errbV = errorFitb(asimvf_big, refit_big, fitaV.intercept_ , fitaV.coef_[0])


erramH = errorFita(momentH_big, zefit_big, fitmH.intercept_ , fitmH.coef_[0])
errbmH = errorFitb(momentH_big, zefit_big, fitmH.intercept_ , fitmH.coef_[0])
erramV = errorFita(momentV_big, refit_big, fitmV.intercept_ , fitmV.coef_[0])
errbmV = errorFitb(momentV_big, refit_big, fitmV.intercept_ , fitmV.coef_[0])




print('The a coefficient for the asymmetry HC is = ', round(fitaH.intercept_,5), '+/-' , round(erraH,5))
print('The b coefficient for the asymmetry HC is = ',  round(fitaH.coef_[0],5), '+/-' , round(errbH,5))

print('The a coefficient for the asymmetry VC is = ',  round(fitaV.intercept_,5), '+/-' , round(erraV,5))
print('The b coefficient for the asymmetry VC is = ',    round(fitaV.coef_[0],5), '+/-' , round(errbV,5))
#print('')
print('################################################################################')
print('################################################################################')
print('')    
print('The a coefficient for the moment HC is = ', round(fitmH.intercept_,5), '+/-' , round(erramH,5))
print('The b coefficient for the moment HC is = ',  round(fitmH.coef_[0],5), '+/-' , round(errbmH,5))

print('The a coefficient for the moment VC is = ',  round(fitmV.intercept_,5), '+/-' , round(erramV,5))
print('The b coefficient for the moment VC is = ',    round(fitmV.coef_[0],5), '+/-' , round(errbmV,5))




xfitR = refit_new
yfitR = (asimvf-overaVfit)/overbVfit

xfitZ = zefit_new
yfitZ = (asimhf-overaHfit)/overbHfit


kR = LinearRegression().fit(xfitR.reshape((-1, 1)), yfitR)
kZ = LinearRegression().fit(xfitZ.reshape((-1, 1)), yfitZ)


print(kZ.intercept_)
print(kZ.coef_)
print('%%%%%%%%%%%%%%%%%%')
print(kR.intercept_)
print(kR.coef_)

plt.figure(1)
plt.scatter(xfitZ, yfitZ)
plt.plot(np.linspace(min(xfitZ),max(xfitZ),100), np.linspace(min(xfitZ),max(xfitZ),100)*kZ.coef_+kZ.intercept_)



plt.figure(2)
plt.scatter(xfitR, yfitR)
plt.plot(np.linspace(min(xfitR),max(xfitR),100), np.linspace(min(xfitR),max(xfitR),100)*kR.coef_+kR.intercept_)



