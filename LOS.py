# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:26:44 2019

@author: sco17122
"""

import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from matplotlib import cm
import math

plt.close('all') 


def N3read(filename):
   inp = open(filename,'r').readlines()
   marker = []
   for i in range(len(inp)):
      if inp[i][:7]=='Channel': marker.append(i+1)
   
   nx = marker[2]-marker[1]-3
   ed = np.zeros(nx+1)
   ny = len(marker)
   N3 = np.zeros((ny, nx))
   
   for i in range(nx+1):
      bits = inp[i+marker[0]].split()
      ed[i] = float(bits[0])

   for i in range(ny):
      for j in range(nx):
         bits = inp[marker[i]+j].split()
         N3[i,j] = float(bits[1])

   return ed, N3

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

R0 = 3.3
Z0 = 0.504
E0 = 1.1
Ra = 2.9
Za = 0.2
Ea = 1.6
R20 = 0.0
R2a = .0
minor = 0.9
alpha = 5.1166

#time resolution
dt  = .5
#detection efficiency
eff = 0.0157
#attenuation
att = 0.92
#number of counts in the fission chamber
kn1 = 9.92E15

ndf = 1.0 
boo =1.0

#here is created the fake neutron emissivity

ed, N3 = N3read('N3__86825')
LOS = loadKN3LOS()
nLOS=np.zeros(19,'i')
pod = np.zeros((19,3000,5))
for ch in range(19):
   #print N3[ch,:].sum()
  for i in range(len(LOS[ch+1][:,2])):
    LOS[ch+1][i,0] = LHpsi( LOS[ch+1][i,2], LOS[ch+1][i,3], R0, Ra, Z0, Za, E0, Ea, minor)
  k = np.where(LOS[ch+1][:,0]<1)  
  n = len(k[0])
  nLOS[ch] = n
  for m in range(n):
    pod[ch,m,0] = LOS[ch+1][k[0][m],0]
    pod[ch,m,1] = LOS[ch+1][k[0][m],4]
    pod[ch,m,2] = LOS[ch+1][k[0][m],2]
    pod[ch,m,3] = LOS[ch+1][k[0][m],3]     
    
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
        print (rho, Rx, Zx, Ex, R2x ,i)
        for j in range(40):
           theta = dtheta * j
           R = Rx + minor * rho * np.cos(theta) + R2x * np.cos(2.*theta)
           Z = Zx + Ex * ( minor * rho * np.sin(theta)-R2x*np.sin(2.*theta))
           k = 40*(i-1)+j
           points[k,:] = (R,Z,(1.0-rho*rho)**alpha+0.01) 
      #     print k, points[k,:]    
tx = np.linspace(1.5, 4.0, 100)
ty = np.linspace(-2.0, 2.0, 100)
XI, YI = np.meshgrid(tx, ty)
# use RBF
rbf = Rbf(points[:,0], points[:,1], points[:,2], epsilon=0.001)
ZI = rbf(XI, YI)
# plot the result
plt.figure()
plt.pcolor(XI, YI, ZI, cmap=cm.jet,vmin = 0.0, vmax=1.0)
plt.title('RBF interpolation - multiquadrics')
plt.xlim(1.5, 4)
plt.ylim(-2, 2)
plt.colorbar() 


fil =  'fateball'
ff = open(fil,'w')
ntot = 0.0
mmm = 0
for i in range (20):
    R = 1.95 + i*0.1
    for j in range(-20,20):
        Z = 0.05 + 0.1 * j
        effr = math.hypot((R-Ra),(Z-Za)/Ea)
        ff.write (str(R)+ ' '+str(Z) + ' ' + str(effr) + ' ' + str(rbf(R,Z) * 0.01 * 2 * np.pi * R)+' \r\n')
        if effr<1.0:
            s = rbf(R,Z) * 0.01 * 2 * np.pi * R
            if s<0: continue
            if s<0: print (R,Z,s)
            mmm += 1
            ntot += s
ff.close()



counts = np.zeros((19))
for j in range(19):
    nn = nLOS[j]
  #  print nn
    pod[j,:nn,4] = rbf(pod[j,:nn,2],pod[j,:nn,3])  * pod[j,:nn,1] * dt * eff * att * kn1 * boo / ntot
    counts[j] = pod[j,:nn,4].sum()


exper = np.array((4096,10446,17951,24735,24315,12904,5682,
                  2280,1033,526,90,580,3990,13372,31759,33511,21845,7898,2920))
xv = np.linspace(1,19,19)

plt.figure()
plt.plot(xv,exper,'r',linewidth=2)
plt.plot(xv,counts,'k',linewidth=2)
plt.xlabel('channel', fontsize=14)
plt.ylabel('counts', fontsize=14)
plt.legend(['KN3','synthetic'])
