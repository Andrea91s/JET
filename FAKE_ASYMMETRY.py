#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:40:46 2020

@author: andrea
"""

import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from matplotlib import cm
import math

def asymmetry(counts):
    a1=[]
    a2=[]
    a4=[]
    a5=[]
    a1 = counts[0] + counts[1] + counts[2] + counts[3]
    a2 = counts[4] + counts[5] + counts[6] + counts[7] + counts[8] + counts[9]
    a4 = counts[10] + counts[11] + counts[12] + counts[13] + counts[14]/2
    a5 = counts[14]/2 + counts[15] + counts[16] + counts[17] + counts[18]  
    asimh = (a1 - a2) / (a1 + a2)
    asimv = (a4 - a5) / (a4 + a5)    
    errasimh = (2*np.sqrt(a1*a2))/(np.sqrt((a1+a2)**3))
    errasimv = (2*np.sqrt(a5*a4))/(np.sqrt((a4+a5)**3))
    return print('Horizontal asymmetry is', round(asimh,4) ,'+-', round(errasimh,4))#, 'and Vertical asymmetry is', round(asimv,4), '+-', round(errasimv,4))

def weight(a):
    wH = np.average(np.linspace(1,10,10), weights = a[0:10])
    wV = np.average(np.linspace(11,19,9), weights = a[10:19])
    sigW=0
    for qqq in range(1, 11):  
        c = a[0:10]
        W = np.average(np.linspace(1,10,10), weights = c)        
        sigW  =  np.sqrt((((qqq*sum(c) - W*sum(c)) / (sum(c))**2)   **2)*np.sqrt(c[qqq-1])) + sigW
    return print('Horizontal weight is', round(wH,4) ,'+-', round(sigW,4))#, 'and Vertical weight is', round(wV,4), '+-', round(errmv,4))



shot = [85393] ; scenario=5
filt=15 ;countsleft=200 ;countsright=500
#for w in range(0, len(shot)):
#    print(shot[w])
#    time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCtime_'+ str(shot[w])+'.txt')
#    f = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCdata_'+ str(shot[w])+'.txt')
#    refita = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/REFIT_'+ str(shot[w])+'.txt')
#    zefita = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ZEFIT_'+ str(shot[w])+'.txt')
#    time_maga = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/time_mag_'+ str(shot[w])+'.txt')
#    fc = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/neut'+ str(shot[w])+'.txt')
#    fc_time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/neut_time'+ str(shot[w])+'.txt')
#exp = np.reshape(f,(len(time),20))/100   #NEW NEUTRON CAMERA

#timeindex = np.sum(exp,1)
#
#idxa = (np.where((timeindex>countsleft)))
#idxb = (np.where((timeindex>countsright)))
#a = time[idxa] 
#b = time[idxb]
#
#t1 = a[0] 
#t2 = b[-1] 
#
#
#
##Here I define the length and the area of collimators
#
#Len   = np.array((  1.615991444 ,  1.581272864 , 1.552087292 ,  1.532928955 , 1.522378565 ,  1.521778565 ,  1.533228955 , 1.553687292 ,  1.581572864 ,  1.617691444 ,  1.518333124 ,  1.515424142 ,  1.511876256 ,  1.510795114 ,  1.5093 ,  1.510695237 ,  1.514264589 ,  1.514830584 ,   1.518823781))
#radii = np.array((        1.050 ,        1.050 ,       1.050 ,        1.050 ,       1.050 ,        1.050 ,        1.050 ,       1.050 ,        1.050 ,        1.050 ,        0.500 ,        0.600 ,        0.750 ,         0.85 ,   1.050 ,        1.050 ,        1.050 ,        1.050 ,         1.050))/100.0 
#areas = np.pi * (radii**2) 
#
# #the horizontal camera has  10 channels (1-10)
## the vertical camera has 9 channels (11-19)
#Ha = np.zeros((len(time),10))
#Va = np.zeros((len(time),9))
#Hchannel = np.linspace(1,10,10)
#Vchannel = np.linspace(11,19,9)
#Ha = exp[0:-1, 0:10]
#Va = exp[0:-1, 10:19]

#plt.close('all') 

def loadKN3LOS():
    LOS = {}
    for i in range(19):
        LOS[i+1] = np.loadtxt("/home/andrea/JET/KN3_"+str(i+1)+"P.out")  
    return LOS



def LHpsi(R,Z,R0,Ra,Z0,Za,e0,ea,a):
    R1 = R-R0
    R2 = Ra-R0
    E =  1.4 
    Z1 = (Z-Z0)/E
    Z2 = (Za-Z0)/E
    A = R2*R2+Z2*Z2-a*a
    B = -2*(R1*R2+Z1*Z2)
    C = R1*R1+Z1*Z1
    inn = B*B-4*A*C
    out = 0.0    
    if inn>0.0 :
        s1 = (-B-np.sqrt(inn))/2.0/A
        s2 = (-B+np.sqrt(inn))/2.0/A
        if 0<=s1: out = s1    
        if 0<=s2: out = s2    
    
    return out    
# 5.01665899  0.03334237  3.22571729  0.40389119  1.48744515

R0 = 3.03 # MAGNETIC AXIS POSITION
Z0 = 0.26 # MAGNETIC AXIS POSITION
E0 = 1.6 # ELLIPTICITY ON THE AXIS
Ra = 3 # MAJOR RADIUS
Za = (27+0)/100 # DISTANCE FROM THE MIDPLANE
Ea = 1.8  # ELLIPTICITY RHO=1
R20 = 0
R2a = 0
minor = 0.9
alpha = 2.1166
alpha_ho = 9.1166



#time resolution
dt  = 0.001
#detection efficiency
eff = 0.0157
#attenuation
att = 0.92
#number of counts in the fission chamber
kn1 = (3.916946001936384e+16)*2

ndf = 1.0 
boo = 1.0

#here is created the fake neutron emissivity


LOS = loadKN3LOS()
nLOS=np.zeros(19,'i')
pod_std = np.zeros((19,3000,5))
pod_ho = np.zeros((19,3000,5))
pod_ban = np.zeros((19,3000,5))
for ch in range(19):
   #print N3[ch,:].sum()
  for i in range(len(LOS[ch+1][:,2])):
    LOS[ch+1][i,0] = LHpsi( LOS[ch+1][i,2], LOS[ch+1][i,3], R0, Ra, Z0, Za, E0, Ea, minor)
  k = np.where(LOS[ch+1][:,0]<1)  
  n = len(k[0])
  nLOS[ch] = n
  for m in range(n):
    pod_std[ch,m,0] = LOS[ch+1][k[0][m],0]
    pod_std[ch,m,1] = LOS[ch+1][k[0][m],4]
    pod_std[ch,m,2] = LOS[ch+1][k[0][m],2]
    pod_std[ch,m,3] = LOS[ch+1][k[0][m],3] 
    pod_ho[ch,m,0] = LOS[ch+1][k[0][m],0]
    pod_ho[ch,m,1] = LOS[ch+1][k[0][m],4]
    pod_ho[ch,m,2] = LOS[ch+1][k[0][m],2]
    pod_ho[ch,m,3] = LOS[ch+1][k[0][m],3] 
    pod_ban[ch,m,0] = LOS[ch+1][k[0][m],0]
    pod_ban[ch,m,1] = LOS[ch+1][k[0][m],4]
    pod_ban[ch,m,2] = LOS[ch+1][k[0][m],2]
    pod_ban[ch,m,3] = LOS[ch+1][k[0][m],3]     
    
stype=1
if stype==0:
    points = np.zeros((20,3))
    theta = np.linspace(0, 2*np.pi,15)
    points[:14,0] = np.cos(theta[:14]) + 2.9
    points[:14,1] = np.sin(theta[:14])*1.4
    points[14,:] = (3, 0.25 , 1)
    points[15,:] = (3.2, 0.25 , .9)
    points[16,:] = (2.8, 0.25 , .9)
    points[17,:] = (3,  0.8 , .9)
    points[18,:] = (3, -0.4 , .1)
    points[19,:] = (3, 1 , .4)
        
if stype == 1:    
    dtheta = np.pi*2.0/40.0
    points = np.zeros((800,3))
    for i in range(1,21):
        rho= i/20.0
        Rx  = R0 + rho * (Ra-R0)
        Zx  = Z0 + rho * (Za-Z0)
        Ex  = E0 + rho * (Ea-E0)
        R2x = R20 + rho * (R2a-R20)   
        #print (rho, Rx, Zx, Ex, R2x ,i)
        for j in range(40):
           theta = dtheta * j
           R = Rx + minor * rho * np.cos(theta) + R2x * np.cos(2.*theta)
           Z = Zx + Ex * ( minor * rho * np.sin(theta)-R2x*np.sin(2.*theta))
           k = 40*(i-1)+j
           points[k,:] = (R,Z,(1.0-rho*rho)**alpha+0.01) 
           
      #     print k, points[k,:]   
points_std = points     
#make hollow profile

stype=1
if stype==0:
    points = np.zeros((20,3))
    theta = np.linspace(0, 2*np.pi,15)
    points[:14,0] = np.cos(theta[:14]) + 2.9
    points[:14,1] = np.sin(theta[:14])*1.4
    points[14,:] = (3, 0.25 , 1)
    points[15,:] = (3.2, 0.25 , .9)
    points[16,:] = (2.8, 0.25 , .9)
    points[17,:] = (3,  0.8 , .9)
    points[18,:] = (3, -0.4 , .1)
    points[19,:] = (3, 1 , .4)
        
if stype == 1:    
    dtheta = np.pi*2.0/40.0
    points_ho = np.zeros((800,3))
    for i in range(1,21):
        rho= i/20.0
        Rx  = R0 + rho * (Ra-R0)
        Zx  = Z0 + rho * (Za-Z0)
        Ex  = E0 + rho * (Ea-E0)
        R2x = R20 + rho * (R2a-R20)   
        #print (rho, Rx, Zx, Ex, R2x ,i)
        for j in range(40):
           theta = dtheta * j
           R = Rx + minor * rho * np.cos(theta) + R2x * np.cos(2.*theta)
           Z = Zx + Ex * ( minor * rho * np.sin(theta)-R2x*np.sin(2.*theta))
           k = 40*(i-1)+j
           points_ho[k,:] = (R,Z,(1.0-rho*rho)**alpha_ho+0.01) 
      #     print k, points[k,:]         

  
tx = np.linspace(1.5, 4.0, 200)
ty = np.linspace(-2.0, 2.0, 200)
XI, YI = np.meshgrid(tx, ty)

rbf_std = Rbf(points_std[:,0], points_std[:,1], points_std[:,2], epsilon=0.001)


rbf_ho = Rbf(points_ho[:,0], points_ho[:,1], points_std[:,2]-points_ho[:,2], epsilon=0.001)
points_ban = points_ho
points_ban[:,2] = points_std[:,2]-points_ho[:,2]
idx = np.where((points_ban[:,0]<R0))
points_ban[idx,2]=0

rbf_ban = Rbf(points_ban[:,0], points_ban[:,1], points_ban[:,2], epsilon=0.001)

# use RBF

ZI_std = rbf_std(XI, YI)
ZI_ho = rbf_ho(XI, YI)
ZI_ban = rbf_ban(XI, YI)



fil =  'fateball_std'
ff = open(fil,'w')
ntot = 0.0
mmm = 0
for i in range (20):
    R = 1.95 + i*0.1
    for j in range(-20,20):
        Z = 0.05 + 0.1 * j
        effr = math.hypot((R-Ra),(Z-Za)/Ea)
        ff.write (str(R)+ ' '+str(Z) + ' ' + str(effr) + ' ' + str(rbf_std(R,Z) * 0.01 * 2 * np.pi * R)+' \r\n')
        if effr<1.0:
            s = rbf_std(R,Z) * 0.01 * 2 * np.pi * R
            if s<0: continue
            if s<0: print (R,Z,s)
            mmm += 1
            ntot += s
ff.close()

fil =  'fateball_std'
ff = open(fil,'w')
ntot = 0.0
mmm = 0
for i in range (20):
    R = 1.95 + i*0.1
    for j in range(-20,20):
        Z = 0.05 + 0.1 * j
        effr = math.hypot((R-Ra),(Z-Za)/Ea)
        ff.write (str(R)+ ' '+str(Z) + ' ' + str(effr) + ' ' + str(rbf_ho(R,Z) * 0.01 * 2 * np.pi * R)+' \r\n')
        if effr<1.0:
            s = rbf_ho(R,Z) * 0.01 * 2 * np.pi * R
            if s<0: continue
            if s<0: print (R,Z,s)
            mmm += 1
            ntot += s
ff.close()

fil =  'fateball_ban'
ff = open(fil,'w')
ntot = 0.0
mmm = 0
for i in range (20):
    R = 1.95 + i*0.1
    for j in range(-20,20):
        Z = 0.05 + 0.1 * j
        effr = math.hypot((R-Ra),(Z-Za)/Ea)
        ff.write (str(R)+ ' '+str(Z) + ' ' + str(effr) + ' ' + str(rbf_ban(R,Z) * 0.01 * 2 * np.pi * R)+' \r\n')
        if effr<1.0:
            s = rbf_ban(R,Z) * 0.01 * 2 * np.pi * R
            if s<0: continue
            if s<0: print (R,Z,s)
            mmm += 1
            ntot += s
ff.close()

counts_std = np.zeros((19))
counts_ho = np.zeros((19))
counts_ban = np.zeros((19))
for j in range(19):
    nn = nLOS[j]
  #  print nn
    pod_std[j,:nn,4] = rbf_std(pod_std[j,:nn,2],pod_std[j,:nn,3])  * pod_std[j,:nn,1] * dt * eff * att * kn1 * boo / ntot
    pod_ho[j,:nn,4] = rbf_ho(pod_ho[j,:nn,2],pod_ho[j,:nn,3])  * pod_ho[j,:nn,1] * dt * eff * att * kn1 * boo / ntot
    pod_ban[j,:nn,4] = rbf_ban(pod_ban[j,:nn,2],pod_ban[j,:nn,3])  * pod_ban[j,:nn,1] * dt * eff * att * kn1 * boo / ntot
    counts_std[j] = pod_std[j,:nn,4].sum()
    counts_ho[j] = pod_ho[j,:nn,4].sum()
    counts_ban[j] = pod_ban[j,:nn,4].sum()
    
    
#---Plot the first wall---
with open('FirstWall_86825.txt', 'r') as f:
    w = f.readlines()
    rcord = np.zeros(len(w))
    zcord = np.zeros(len(w))
    for i in range(0, len(w)):
        temp1, temp2 = w[i].split()
        rcord[i] = float(temp1)
        zcord[i] = float(temp2)
    rcord = rcord*0.01 #Change to units of meters
    zcord = zcord*0.01 #Change to units of meters

    

xv = np.linspace(1,19,19)

coeff = np.loadtxt('LOS_coefficients_KN3.txt')


plt.close('all')


plt.figure(1,figsize = (13, 8))
plt.subplot(1,2,1)
for i in range(0,10):
    plt.plot(np.linspace(1.8,4, 100), np.linspace(1.8,4, 100)*coeff[i,0] + coeff[i,1],'k')
plt.text(1.74, 1.49,'1',verticalalignment='top', horizontalalignment='right', fontsize=10)
plt.text(1.74, -1.47,'10',verticalalignment='top', horizontalalignment='right', fontsize=10) 
plt.text(1.74, -1.82,'11',verticalalignment='top', horizontalalignment='right', fontsize=10)
plt.text(3.96, -1.06,'19',verticalalignment='top', horizontalalignment='right', fontsize=10) 
    
#plt.plot(3.0215*np.ones((1000)), np.linspace(-2 ,4, 1000),'k')
plt.pcolor(XI, YI, ZI_std, cmap=cm.hot_r,vmin = 0.0, vmax=1.0)
plt.plot(rcord[:], zcord[:], 'r')    
plt.plot(R0,Z0,'r+', markersize=20)
#plt.title('RBF interpolation - multiquadrics')
plt.xlim(1.5, 4)
plt.ylim(-2, 2.5)
plt.xlabel('R (m)', fontsize=14)
plt.ylabel('Z (m)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax4=plt.subplot(1,2,2)

plt.plot(xv,counts_std,'k',linewidth=2)
plt.plot(xv,counts_std,'r+',markersize=14)
#plt.plot(Hchannel,np.sum(Ha,0)/10000,'r',linewidth=2)
#plt.plot(Vchannel,np.sum(Va,0)/10000,'r',linewidth=2)
plt.xlabel('channel', fontsize=14)
plt.ylabel('counts ', fontsize=14)
#plt.legend(['Syntethic','KN3'],fontsize=8,frameon=False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.ylim(0,55)


#
#plt.figure(1,figsize = (13, 8))
#plt.subplot(2,3,1)
#for i in range(0,19):
#    plt.plot(np.linspace(1.8,4, 1000), np.linspace(1.8,4, 1000)*coeff[i,0] + coeff[i,1],'k')
#plt.text(1.74, 1.49,'1',verticalalignment='top', horizontalalignment='right', fontsize=10)
#plt.text(1.74, -1.47,'10',verticalalignment='top', horizontalalignment='right', fontsize=10) 
#plt.text(1.74, -1.82,'11',verticalalignment='top', horizontalalignment='right', fontsize=10)
#plt.text(3.96, -1.06,'19',verticalalignment='top', horizontalalignment='right', fontsize=10) 
#    
#plt.plot(3.0215*np.ones((1000)), np.linspace(-2 ,4, 1000),'k')
#plt.pcolor(XI, YI, ZI_std, cmap=cm.hot_r,vmin = 0.0, vmax=1.0)
#plt.plot(rcord[:], zcord[:], 'r')    
#plt.plot(R0,Z0,'r+', markersize=20)
##plt.title('RBF interpolation - multiquadrics')
#plt.xlim(1.5, 4)
#plt.ylim(-2, 2.5)
#plt.xlabel('R (m)', fontsize=14)
#plt.ylabel('Z (m)', fontsize=14)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#
#ax2=plt.subplot(2,3,2)
#for i in range(0,19):
#    plt.plot(np.linspace(1.8,4, 1000), np.linspace(1.8,4, 1000)*coeff[i,0] + coeff[i,1],'k')
#plt.text(1.74, 1.49,'1',verticalalignment='top', horizontalalignment='right', fontsize=10)
#plt.text(1.74, -1.47,'10',verticalalignment='top', horizontalalignment='right', fontsize=10) 
#plt.text(1.74, -1.82,'11',verticalalignment='top', horizontalalignment='right', fontsize=10)
#plt.text(3.96, -1.06,'19',verticalalignment='top', horizontalalignment='right', fontsize=10) 
#
#plt.plot(3.0215*np.ones((1000)), np.linspace(-2 ,4, 1000),'k')
#plt.pcolor(XI, YI, ZI_ho, cmap=cm.hot_r,vmin = 0.0, vmax=1.0)
#plt.plot(rcord[:], zcord[:], 'r')    
#plt.plot(R0,Z0,'r+', markersize=20)
##plt.title('RBF interpolation - multiquadrics')
#plt.xlim(1.5, 4)
#plt.ylim(-2, 2.5)
#plt.xlabel('R (m)', fontsize=14)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#plt.setp(ax2.get_yticklabels(), visible=False)
#
#ax3=plt.subplot(2,3,3)
#for i in range(0,19):
#    plt.plot(np.linspace(1.8,4, 1000), np.linspace(1.8,4, 1000)*coeff[i,0] + coeff[i,1],'k')
#plt.text(1.74, 1.49,'1',verticalalignment='top', horizontalalignment='right', fontsize=10)
#plt.text(1.74, -1.47,'10',verticalalignment='top', horizontalalignment='right', fontsize=10) 
#plt.text(1.74, -1.82,'11',verticalalignment='top', horizontalalignment='right', fontsize=10)
#plt.text(3.96, -1.06,'19',verticalalignment='top', horizontalalignment='right', fontsize=10) 
#plt.plot(3.0215*np.ones((1000)), np.linspace(-2 ,4, 1000),'k')
#plt.pcolor(XI, YI, ZI_ban, cmap=cm.hot_r,vmin = 0.0, vmax=1.0)
#plt.plot(rcord[:], zcord[:], 'r')    
#plt.plot(R0,Z0,'r+', markersize=20)
##plt.title('RBF interpolation - multiquadrics')
#plt.xlim(1.5, 4)
#plt.ylim(-2, 2.5)
#plt.xlabel('R (m)', fontsize=14)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#
#plt.setp(ax3.get_yticklabels(), visible=False)
#
#ax4=plt.subplot(2,3,4)
#
#plt.plot(xv,counts_std/10000,'k',linewidth=2)
#plt.plot(xv,counts_std/10000,'r+',markersize=14)
##plt.plot(Hchannel,np.sum(Ha,0)/10000,'r',linewidth=2)
##plt.plot(Vchannel,np.sum(Va,0)/10000,'r',linewidth=2)
#plt.xlabel('channel', fontsize=14)
#plt.ylabel('counts ($x 10^4$)', fontsize=14)
##plt.legend(['Syntethic','KN3'],fontsize=8,frameon=False)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#plt.ylim(0,55)
#
#
#ax5 = plt.subplot(2,3,5, sharex = ax4)
#
#plt.plot(xv,counts_ho/10000,'k',linewidth=2)
#plt.plot(xv,counts_ho/10000,'r+',markersize=14)
##plt.plot(Hchannel,np.sum(Ha,0)/10000,'r',linewidth=2)
##plt.plot(Vchannel,np.sum(Va,0)/10000,'r',linewidth=2)
#plt.xlabel('channel', fontsize=14)
##plt.legend(['Syntethic','KN3'],fontsize=8,frameon=False)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#plt.ylim(0,55)
#plt.setp(ax5.get_yticklabels(), visible=False)
#
#ax6 = plt.subplot(2,3,6, sharex = ax4)
#
#
#plt.plot(xv,counts_ban/10000,'k',linewidth=2)
#plt.plot(xv,counts_ban/10000,'r+',markersize=14)
##plt.plot(Hchannel,np.sum(Ha,0)/10000,'r',linewidth=2)
##plt.plot(Vchannel,np.sum(Va,0)/10000,'r',linewidth=2)
#plt.xlabel('channel', fontsize=14)
##plt.legend(['Syntethic','KN3'],fontsize=8,frameon=False)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#plt.ylim(0,55)
#
#plt.setp(ax6.get_yticklabels(), visible=False)



#ASYMMETRY

asymmetry(counts_std)
#asymmetry(counts_ho)
#asymmetry(counts_ban)

#WA

weight(counts_std)
#weight(counts_ho)
#weight(counts_ban)
