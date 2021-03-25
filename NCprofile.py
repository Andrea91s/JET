# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:04:10 2020

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def interpolation(newtime, oldtime, array):
    return np.interp(newtime, oldtime, array)




shot = [94665]
filt=451 ;countsleft=200 ;countsright=200



for w in range(0, len(shot)):
    #plt.close('all')
    print(shot[w])
    
    time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCtime_'+ str(shot[w])+'.txt')
    f = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCdata_'+ str(shot[w])+'.txt')

      
   
#    newtime = np.linspace(round(float((time[0]))), round(float((time[-1]))), (round(float((time[-1])))-round(float((time[0]))))*1000)
#
#        
#    counts_old = np.reshape(f,(len(time),20))   #NEW NEUTRON CAMERA
#    counts = np.zeros((len(newtime),20))
#    
#    if round(float(np.diff(time)[0]*1000)) > 1.5:
#        for i in range(len(counts_old[1])):
#            counts[:,i] = round(float(np.diff(time)[0]*1000))*(interpolation(newtime,time,counts_old[:,i])/1000)
#        time=newtime
#    else:
#        counts = np.zeros((len(time),20))
#        counts = counts_old
#        print('The time resolution is 1 ms, please go ahead with the analysis.')    
#        
    #I DIVIDED BY 100 because the integration time is 10 ms
    counts = np.reshape(f,(len(time),20))/100   #NEW NEUTRON CAMERA
    #counts = np.reshape(f,(len(time),19))/100  #OLD NEUTRON CAMERA

# Here I select one the entries with number of counts greater than a certain number, both for left ansd right time limit

    timeindex = np.sum(counts,1)

    idxa = (np.where((timeindex>20)))
    idxb = (np.where((timeindex>20)))
    a = time[idxa] 
    b = time[idxb]

    t1 = a[0] 
    t2 = b[-1] 

    t1=49.3
    t2=49.4
    
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
    test = Ha
    #Ha[:,0] = test[:,5]
    #Ha[:,5] = test[:,0]
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

    H = H[idx]
    V = V[idx]
    
    Hfake = np.multiply(H,areas[0:10]**2) /(Len[0:10]**2)
    Vfake = np.multiply(V,areas[10:19]**2) / (Len[10:19]**2)  
    asimv = np.zeros((len(t),9))
    a4 = np.zeros((len(t)))
    a5 = np.zeros((len(t)))
    for k in range(0, len(t)):
        a4[k] = V[k,0] + V[k,1] + V[k,2] + V[k,3] + V[k,4]/2 
        a5[k] = V[k,4]/2 + V[k,5] + V[k,6] + V[k,7] + V[k,8]    
        asimv = (a4 - a5) / (a4 + a5) 
    idx2 = (np.where((t>47.61) & (t<48.2)))
    idx2 = (np.where((t>49.045) & (t<49.537)))
    idx2 = (np.where((t>50.46) & (t<50.63)))
    idx2 = (np.where((t>51.92) & (t<52.27)))
    twind=t[idx2]
    Vwind = V[idx2]
    
    plt.figure(1)
    plt.plot(Vchannel,np.sum(Vfake,0))
    plt.xlabel('LOS channel')
    plt.ylabel('Arb')
    plotc = np.linspace(11,19,9)
    plt.xticks(plotc)
    plt.title(str(shot[w]))  
    #np.savetxt('/home/andrea/PAPERS/JETREALTIME/Figure4/t3.dat',np.mean(Vwind,0))
    asimv = np.zeros((len(twind),9))
    a4 = np.zeros((len(twind)))
    a5 = np.zeros((len(twind)))
    for k in range(0, len(twind)):
        a4[k] = Vwind[k,0] + Vwind[k,1] + Vwind[k,2] + Vwind[k,3] + Vwind[k,4]/2 
        a5[k] = Vwind[k,4]/2 + Vwind[k,5] + Vwind[k,6] + Vwind[k,7] + Vwind[k,8]    
        asimv = (a4 - a5) / (a4 + a5)     
    
#    for i in range(len(Hfake)):
#     plt.figure(1)
#     plt.plot(Hchannel,Hfake.T,'red')
#     plt.plot(Vchannel,Vfake[i,:])
#     plt.xlabel('LOS channel')
#     plt.ylabel('Arb')
#     plotc = np.linspace(11,19,9)
#     plt.xticks(plotc)
#     plt.title(str(shot[w]))  
#    [T, CH] = np.meshgrid(t,Vchannel)  
#    
#    spec=np.zeros((9, len(t)))
#    for wf in range(len(t)):
#        spec[:,wf] = V[wf,:]/V[0,:]
#    #spec = spec.transpose().copy()
#    spec[np.where(spec==0)] = 1    
#    plt.figure(1)
#    plt.contourf(T, CH, spec, cmap = 'jet', vmin=0.5, vmax=1.5)
#    plt.colorbar()
#    plt.xlabel('LOS channel')
#    plt.ylabel('Arb')
#    plt.xlim(t1, t2)
#    plt.title(str(shot[w]))  
     
#
#    plt.plot(Hchannel,np.sum(Hfake,0),'red')
#    plt.plot(Vchannel,np.sum(Vfake,0),'blue')
#    plt.scatter(Hchannel,np.sum(Hfake,0),c="k")
#    plt.scatter(Vchannel,np.sum(Vfake,0),c="k")
#    plt.xlabel('LOS channel')
#    plt.ylabel('Arb')
#    plotc = np.linspace(1,19,19)
#    plt.xticks(plotc)
#    plt.title(str(shot[w]))   
#    plt.savefig('test_' + str(shot[w]) + '.png')
#plt.figure(2)
#plt.plot(twind,signal.savgol_filter(asimv,15,1))