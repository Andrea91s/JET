#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:52:20 2020

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt


bigR=[]
bigZ=[]

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


#shot=[82307,82575,82579,82828,82829,82831,82833,82859,82861,82862,82866,82874,82877,82878,82878,82880,82883,82894,82895,82896,83180,83182,83183, 
#      84501,84502,84503, 84504,84506,84507,84511,84512,84513, 84514, 84515, 84516, 84517, 84519, 86682, 86683, 86684, 86685, 86686, 86687, 86688, 86888,
#      96058, 96059]
#shot = [94217,94416,94417,94635,96435,96664,96665,96730,96768,96777,96947,96994,97503]
zzshot=[82812]
plt.close('all')
for w in range(0, len(shot)):
    #plt.close('all')
    print(shot[w])
    neut_time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/neut_time'+ str(shot[w])+'.txt')
    neut = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/neut'+ str(shot[w])+'.txt')
    nbi_time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/nbi_time'+ str(shot[w])+'.txt')
    nbi = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/nbi'+ str(shot[w])+'.txt')
    #icrh_time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/icrh_time'+ str(shot)+'.txt')
    #icrh = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/icrh'+ str(shot)+'.txt')
    #ip_time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/ip_time'+ str(shot[w])+'.txt')
    #ip = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/ip'+ str(shot[w])+'.txt')
    t_time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/t_time_'+ str(shot[w])+'.txt')
    t = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/t'+ str(shot[w])+'.txt')
    refit = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/REFIT_'+ str(shot[w])+'.txt')
    zefit = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ZEFIT_'+ str(shot[w])+'.txt')
    time_maga = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/time_mag_'+ str(shot[w])+'.txt')
    dens_time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/dens_time'+ str(shot[w])+'.txt')
    dens = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/dens'+ str(shot[w])+'.txt')
    nc_time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCtime_'+ str(shot[w])+'.txt')
    f = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCdata_'+ str(shot[w])+'.txt')
    
    counts = np.reshape(f,(len(nc_time),20))/100   #NEW NEUTRON CAMERA
    
    t1 = 40
    t2 = 60
    idx = (np.where((time_maga>t1) & (time_maga<t2)))
    time_maga = time_maga[idx] 
    refita = refit[idx]
    zefita = zefit[idx]
 
    if any(i<=2.95 for i in refita) is True:
       bigR=np.append(bigR,shot[w])
     
    if any(k>=0.26 for k in zefita) is True:
       bigZ=np.append(bigZ,shot[w])  


    
#    plt.figure(1)
#    plt.subplot(2,1,1)
#    plt.legend(fontsize = 8, ncol=1)
#    plt.plot(time_maga, refita, label='%s' % str(shot[w]))
#    plt.ylabel('R (m)')
#    plt.xlim(t1, t2)
#    plt.subplot(2,1,2)
#    plt.legend(fontsize = 8, ncol=1)
#    plt.plot(time_maga, zefita, label='%s' % str(shot[w]))
#    plt.ylabel('Z (m)')
#    plt.xlim(t1, t2)
#    
    plt.figure(1)
    plt.subplot(4,1,1)
    plt.legend(fontsize = 8, ncol=1)
    plt.plot(neut_time, neut, label='%s' % str(shot[w]))
    plt.ylabel('neutrons/s')
    plt.xlim(t1, t2)
    plt.subplot(4,1,2)
    plt.plot(nbi_time,nbi)
    plt.ylabel('NBI W')
    plt.xlim(t1, t2)
    plt.subplot(4,1,3)
    plt.plot(t_time, t/1000)
    plt.ylabel('Te (keV)')    
    plt.xlim(t1, t2)    
    plt.subplot(4,1,4)
    plt.plot(dens_time, dens)
    plt.ylabel('ne (m^-3)')
    plt.xlim(t1, t2)
    plt.xlabel('time (s)')
##    
#    
#  
#    plt.figure(3)
#    plt.subplot(3,1,1)
#    plt.legend(fontsize = 8, ncol=1)
#    plt.plot(nc_time, np.sum(counts, axis=1), label='%s' % str(shot[w]))
#    plt.ylabel('neutrons/s')
#    plt.xlim(t1, t2)
#    plt.subplot(3,1,2)
#    plt.plot(time_maga, refita)
#    plt.ylabel('R (m)')    
#    plt.xlim(t1, t2)
#    plt.subplot(3,1,3)
#    plt.plot(time_maga, zefita)
#    plt.ylabel('Z (m)')    
#    plt.xlim(t1, t2)
#
#    
#
#   
#    
#    
#    print('neutron camera integration time is',round((np.diff(nc_time)*1000)[1]),'ms')
#    print('efit integration time is',round((np.diff(time_maga)*1000)[1]),'ms')
#   