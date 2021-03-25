# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:04:10 2020

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression


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

superbig_AH_a = []
superbig_AH_b = []
superbig_AV_a = []
superbig_AV_b = []
superbig_MH_a = []
superbig_MH_b = []
superbig_MV_a = []
superbig_MV_b = []
errsuperbig_AH_a = []
errsuperbig_AH_b = []
errsuperbig_AV_a = []
errsuperbig_AV_b = []
errsuperbig_MH_a = []
errsuperbig_MH_b = []
errsuperbig_MV_a = []
errsuperbig_MV_b = []
#
#shot=[84536,84539, 84540, 84541,84542,84543,84544,84545,84786,84788,84789,84792,84796,84797,84798,
#      85197, 85201, 85203, 85204, 85205,85393, 85394, 85399,85400, 85401, 85402, 85403,
#      92393, 92394, 92395, 92398, 92399,94256,94257,94259,94261,94262,94263,94264,94265,94266,94268,94270,
#      86658, 86659,86660,86661,86662,86663,86664,86667,86668,86669,86670,86671,86672,86673,86674,86675,86676,86677,86678,86679,
#      94665,94666,94667,94669,94670,94671,94672,94674,94675,94676,94677,94678,94679,94680,94681,94682,94683,94684,
#      85228	,85229	,85230	,85231	,85232	,86362	,86365	,86366	,86367	,86368	,86374	,86375	,86377	,
#      86378	,86379	,86380	,86381	,86382	,86383	,86386	,86388	,86389	,86390	,86392,	86399	,86401	,86404	,86407	,
#      86411	,86412	,86413	,86414	,86416	,86417	,86419	,86421	,86435	,86436	,86437	,86439	,86441	,86444	,86445,
#      86598	,86599	,86600	,86602	,86603	,86604	,86605	,86606,86982	,86983	,86985	,86986	,86987	,86988	,86989	,86991	,86992	,
#      86993	,86994, 87140	,87142	,87143	,87144,	87160	,87161	,87162	,87164,87228	,87231	,87232	,87233	,87240,87243	,87244	,87246	,
#      87247	,87248	,87249	,87250	,87251	,87252	,87253	,87267	,87268	,87271	,87273	,87276	,87277	,87278	,87280,
#      ]



shot=[82307,82312,82338,82342,82551,82552,82553,82571,82572,82575,82578,82579,82730,82758,82807,82810,82813,82814,82815,82816,82817,82818,
      82820,82821,82828,82829,82831,82833,82859,82861,82862,82863,82866,82870,82870,82872,82874,82876,82877,82878,82878,82880,82883,82894,
      82895,82896,82937,83014,83042,83043,83044,83045,83180,83181,83182,83183,83187,83188,83618, 83631,83633,84495,84496,84497,84498,84499,
      84500,84501,84502,84503, 84504,84506,84507,84508,84509,84510,84511,84512,84513, 84514, 84515, 84516, 84517, 84519,
      84536,84538,84539, 84540, 84541,84542,84543,84544,84545,84786,84787,84788,84789,84792,84793,84794,84795,84796,84797,84798,
      84801,84802,84803, 84804,84806,84808,85100,85185,85186,85187,85192,85197, 85200, 85201, 85203, 85204, 85205,85208,85209,85300,
      85302,85303, 85305,85306,85307,85308,85309,85310,85311,85312,85313,85314,85315,85316,85317,85318,85319,
      85393, 85394, 85398, 85399,85400, 85401,85402,85403, 85406, 85407, 85408, 85409, 86372,86456,86459,86461, 86464, 86614, 86658,
      86659, 86660, 86661,86662,86663,86664,86667,86668,86669, 86670, 86671, 86672, 86673, 86674, 86675,  86676, 86677, 86678, 86679, 86682,
      86683, 86684, 86685, 86686, 86687, 86688, 86774,86775,86825, 86826,86888, 88305, 88307, 88309, 88310, 88311,88312, 88312, 88315, 89301, 
      89302, 89303, 89305, 89306, 89307, 89308, 89309,89310, 89312, 89313, 89314, 89317, 89318, 89319, 90316, 90318, 90319, 90363, 90386,
      91301, 92205, 92207, 92211,92213,92214, 92351, 92380, 92391, 92393, 92394, 92395, 92396, 92398, 92399,92411, 92412, 92413, 92414, 92431,
      94209, 94217, 94243, 94249, 94250, 94251, 94252, 94253, 94256,94257,94259,94261,94262,94263,94264,94265,94266,94267, 94268,94270,
      94271, 94272, 94310, 94323, 94417, 94437, 94441, 94606, 94607, 94610, 94611, 94635,94665, 94666, 94667, 94669, 94670, 94671, 94672, 94674,
      94675, 94676, 94677, 94678, 94679, 94680, 94681, 94682, 94683, 94684, 94701, 95335, 95649, 95650, 95652, 95655, 95665, 95666, 95667, 95668, 
      95669, 95671, 95673, 95674, 95675, 95677, 95679, 95680, 95682, 95683, 95684, 95686, 95688, 95689, 95690, 95691, 95692, 95693, 95694, 95697,
      95698,95699,96058, 96059]
#shot=[92393, 92394, 92395, 92398, 92399,94256,94257,94259,94261,94262,94263,94264,94265,94266,94268,94270,
 #     94665,94666,94667,94669,94670,94671,94672,94674,94675,94676,94677,94678,94679,94680,94681,94682,94683,94684]
shot = np.sort(shot)

filt=351; countsleft=200 ; countsright=200

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
    #print('The integration time of the NC is', round(float(np.diff(time)[0]*1000)),'ms')
    
   
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
#    #counts = np.reshape(f,(len(time),19))/100  #OLD NEUTRON CAMERA

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
    
    zefit_big = zefit_new
    refit_big = refit_new
    momentH_big = momentH
    momentV_big = momentV
    asimhf_big = asimhf
    asimvf_big = asimvf
    errasimh_big = errasimh
    errasimv_big = errasimv
    errmomh_big = errmomh
    errmomv_big = errmomv
    
    
 
  


    ##### THOSE ARE FOR THE ASYMMETRY #######
    indexh = (np.where((errasimh_big==0)))
    indexv = (np.where((errasimv_big==0)))

    errasimh_big[indexh] = np.mean(errasimh_big)
    errasimv_big[indexv] = np.mean(errasimv_big)

    
    fitaH = LinearRegression().fit(zefit_big.reshape((-1, 1)), asimhf_big, sample_weight = 1/errasimh_big**2)
    fitaV = LinearRegression().fit(refit_big.reshape((-1, 1)), asimvf_big, sample_weight = 1/errasimv_big**2)

    fitmH = LinearRegression().fit(zefit_big.reshape((-1, 1)), momentH_big, sample_weight = 1/errmomh_big**2)
    fitmV = LinearRegression().fit(refit_big.reshape((-1, 1)), momentV_big, sample_weight = 1/errmomv_big**2)



    erraH = errorFita(asimhf_big, zefit_big, fitaH.intercept_ , fitaH.coef_[0])
    errbH = errorFitb(asimhf_big, zefit_big, fitaH.intercept_ , fitaH.coef_[0])
    erraV = errorFita(asimvf_big, refit_big, fitaV.intercept_ , fitaV.coef_[0])
    errbV = errorFitb(asimvf_big, refit_big, fitaV.intercept_ , fitaV.coef_[0])


    erramH = errorFita(momentH_big, zefit_big, fitmH.intercept_ , fitmH.coef_[0])
    errbmH = errorFitb(momentH_big, zefit_big, fitmH.intercept_ , fitmH.coef_[0])
    erramV = errorFita(momentV_big, refit_big, fitmV.intercept_ , fitmV.coef_[0])
    errbmV = errorFitb(momentV_big, refit_big, fitmV.intercept_ , fitmV.coef_[0])



#
#    print('The a coefficient for the asymmetry HC is = ', round(fitaH.intercept_,5), '+/-' , round(erraH,5))
#    print('The b coefficient for the asymmetry HC is = ',  round(fitaH.coef_[0],5), '+/-' , round(errbH,5))
#    
#    print('The a coefficient for the asymmetry VC is = ',  round(fitaV.intercept_,5), '+/-' , round(erraV,5))
#    print('The b coefficient for the asymmetry VC is = ',    round(fitaV.coef_[0],5), '+/-' , round(errbV,5))
#    print('')
#    print('################################################################################')
#    print('################################################################################')
#    print('')    
#    print('The a coefficient for the moment HC is = ', round(fitmH.intercept_,5), '+/-' , round(erramH,5))
#    print('The b coefficient for the moment HC is = ',  round(fitmH.coef_[0],5), '+/-' , round(errbmH,5))
#
#    print('The a coefficient for the moment VC is = ',  round(fitmV.intercept_,5), '+/-' , round(erramV,5))
#    print('The b coefficient for the moment VC is = ',    round(fitmV.coef_[0],5), '+/-' , round(errbmV,5))

    superbig_AH_a = np.append(superbig_AH_a, fitaH.intercept_)
    superbig_AH_b = np.append(superbig_AH_b, fitaH.coef_)
    superbig_AV_a = np.append(superbig_AV_a, fitaV.intercept_)
    superbig_AV_b = np.append(superbig_AV_b, fitaV.coef_)
    superbig_MH_a = np.append(superbig_MH_a, fitmH.intercept_)
    superbig_MH_b = np.append(superbig_MH_b, fitmH.coef_)
    superbig_MV_a = np.append(superbig_MV_a, fitmV.intercept_)
    superbig_MV_b = np.append(superbig_MV_b, fitmV.coef_)

    errsuperbig_AH_a = np.append(errsuperbig_AH_a, erraH)
    errsuperbig_AH_b = np.append(errsuperbig_AH_b, errbH)
    errsuperbig_AV_a = np.append(errsuperbig_AV_a, erraV)
    errsuperbig_AV_b = np.append(errsuperbig_AV_b, errbV)
    errsuperbig_MH_a = np.append(errsuperbig_MH_a, erramH)
    errsuperbig_MH_b = np.append(errsuperbig_MH_b, errbmH)
    errsuperbig_MV_a = np.append(errsuperbig_MV_a, erramV)
    errsuperbig_MV_b = np.append(errsuperbig_MV_b, errbmV)
    
    
#np.savetxt('wCOEFF_SHOT.txt',shot)  
#np.savetxt('wCOEFF_ASYM_H_A.txt',superbig_AH_a)  
#np.savetxt('wCOEFF_ASYM_H_B.txt',superbig_AH_b)    
#np.savetxt('wCOEFF_ASYM_V_B.txt',superbig_AV_b)  
#np.savetxt('wCOEFF_ASYM_V_A.txt',superbig_AV_a) 
#np.savetxt('wCOEFF_WEIG_H_A.txt',superbig_MH_a)
#np.savetxt('wCOEFF_WEIG_H_B.txt',superbig_MH_b) 
#np.savetxt('wCOEFF_WEIG_V_B.txt',superbig_MV_b)
#np.savetxt('wCOEFF_WEIG_V_A.txt',superbig_MV_a) 
#np.savetxt('wCOEFF_ERR_ASYM_H_A.txt',errsuperbig_AH_a)
#np.savetxt('wCOEFF_ERR_ASYM_H_B.txt',errsuperbig_AH_b)
#np.savetxt('wCOEFF_ERR_ASYM_V_B.txt',errsuperbig_AV_b)
#np.savetxt('wCOEFF_ERR_ASYM_V_A.txt',errsuperbig_AV_a) 
#np.savetxt('wCOEFF_ERR_WEIG_H_A.txt',errsuperbig_MH_a)
#np.savetxt('wCOEFF_ERR_WEIG_H_B.txt',errsuperbig_MH_b)
#np.savetxt('wCOEFF_ERR_WEIG_V_B.txt',errsuperbig_MV_b) 
#np.savetxt('wCOEFF_ERR_WEIG_V_A.txt',errsuperbig_MV_a)






 