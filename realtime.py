# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:04:10 2020

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


plt.close('all') 

shot = 92380
filt = 5
k=1

#plt.close('all')
time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCtime_'+ str(shot)+'.txt')
f = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCdata_'+ str(shot)+'.txt')
refita = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/REFIT_'+ str(shot)+'.txt')
zefita = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ZEFIT_'+ str(shot)+'.txt')
time_maga = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/time_mag_'+ str(shot)+'.txt')
counts = np.reshape(f,(len(time),20))/100

# Here I select one the entries with number of counts greater than 100

timeindex = np.sum(counts,1)

idx = (np.where((timeindex>300)))
a = time[idx] 

t1 = a[0]
t2 = a[-1]

#Here I define the length and the area of collimators

Len   = np.array((  1.615991444 ,  1.581272864 , 1.552087292 ,  1.532928955 , 1.522378565 ,  1.521778565 ,  1.533228955 , 1.553687292 ,  1.581572864 ,  1.617691444 ,  1.518333124 ,  1.515424142 ,  1.511876256 ,  1.510795114 ,  1.5093 ,  1.510695237 ,  1.514264589 ,  1.514830584 ,   1.518823781))
radii = np.array((        1.050 ,        1.050 ,       1.050 ,        1.050 ,       1.050 ,        1.050 ,        1.050 ,       1.050 ,        1.050 ,        1.050 ,        0.500 ,        0.600 ,        0.750 ,         0.85 ,   1.050 ,        1.050 ,        1.050 ,        1.050 ,         1.050))/100.0 
areas = np.pi * radii * radii 

 #the horizontal camera has  10 channels (1-10)
# the vertical camera has 9 channels (11-19)
Ha = np.zeros((len(time),10))
Va = np.zeros((len(time),9))
Hchannel = np.linspace(1,10,10)
Vchannel = np.linspace(11,19,9)
Ha = counts[0:-1, 0:10]
Va = counts[0:-1, 10:19]

H = np.multiply(Ha,Len[0:10]**2) / (areas[0:10]**2)
V = np.multiply(Va,Len[10:19]**2) / (areas[10:19]**2)

# HERE I SELECT ONLY THE ELEMENTS IN THE TIME WINDOW

idx = (np.where((time>t1) & (time<t2)))
t= time[idx]
idx_efit = (np.where((time_maga>t1) & (time_maga<t2)))
time_mag = time_maga[idx_efit]
refit = refita[idx_efit]
zefit = zefita[idx_efit]
H = H[idx]
V = V[idx]

plt.figure(100)
plt.plot(Hchannel,np.sum(H,0),'red')
plt.plot(Vchannel,np.sum(V,0),'blue')
plt.xlabel('LOS channel')
plt.ylabel('Counts')
plt.title(str(shot))
plt.show()


momentH = np.zeros(len(t))
momentV = np.zeros(len(t))
numbercounts = np.zeros(len(t))
all_zeros = np.zeros(len(t))
     
        
#IF THERE ARE ARRAYS WITH ALL ZERO ELEMENTS I FILL THEM WITH THE PREVIOUS NUMBER OF COUNTS

for i in range(0, len(momentH)):
    all_zeros[i] = not np.any(H[i,:])
    if all_zeros[i] == True:
            H[i,:] = H[i-1,:]

for i in range(0, len(momentV)):
    all_zeros[i] = not np.any(V[i,:])
    if all_zeros[i] == True:
            V[i,:] = V[i-1,:]
                
        
# HERE I CALCULATE THE FIRST MOMENT        
        

for i in range(0, len(momentH)):
    momentH[i] = np.average(Hchannel, weights = H[i,:])
    momentV[i] = np.average(Vchannel, weights = V[i,:])
    numbercounts[i]  = np.sum(H[i,:])  


# THE OBTAINED SIGNAL IS CRAP SO I SMOOTH IT WITH A SAVITZKY-GOLAY FILTER

filtr=signal.savgol_filter(momentV, filt, 3)
filtz=signal.savgol_filter(momentH, filt, 3)

## HERE I CALCULATE THE ASYMMETRY   

a1 = np.zeros(len(t))
a2 = np.zeros(len(t))
a3 = np.zeros(len(t))
a4 = np.zeros(len(t))
a5 = np.zeros(len(t))
a6 = np.zeros(len(t))
#this is calcolated according to https://iopscience.iop.org/article/10.1088/1748-0221/14/09/C09001/pdf

for i in range(0, len(t)):
    a1[i] = H[i,0] + H[i,1] + H[i,2] + H[i,3] 
    a2[i] = H[i,4] + H[i,5] + H[i,6] + H[i,7] + H[i,8] + H[i,9]
    a3[i] = a1[i]+a2[i]
    a4[i] = V[i,0] + V[i,1] + V[i,2] + V[i,3] + V[i,4]
    a5[i] = V[i,5] + V[i,6] + V[i,7] + V[i,8]
    a6[i] = a4[i]+a5[i]
   

asimh = k*((a1 - a2) / a3)
asimv = k*((-a4 + a5) / a6)

asimhf=signal.savgol_filter(asimh, filt, 3)
asimvf=signal.savgol_filter(asimv, filt, 3)

#sigmaaH = (((2 *(a1-a2)**2) / (a3)**3))*0.5
#sigmaaV = (((20 *(a4-a5)**2) / (a6)**3))*0.5


fig = plt.figure(figsize = (23, 17))
fig.suptitle(str(shot), fontsize=26)

ax=plt.subplot(2,2,1)
ax.set_xlabel('time (s)', fontsize=14)
ln1 = ax.plot(time_mag, zefit,'k', linewidth=2,label = 'Z EFIT')
ax.set_ylabel('Z (m)', fontsize=14)
ax2=ax.twinx()
ln2 = ax2.plot(t, filtz,'r', linewidth=2,label = 'NCH')
ax2.set_ylabel('First moment', fontsize=14)
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0, frameon=False)



ax=plt.subplot(2,2,2)
ax.set_xlabel('time (s)', fontsize=14)
ln1 = ax.plot(time_mag, refit,'k', linewidth=2,label = 'R EFIT')
ax.set_ylabel('R (m)', fontsize=14)
ax2=ax.twinx()
ln2 = ax2.plot(t, filtr,'r', linewidth=2,label = 'NCV')
ax2.set_ylabel('First moment', fontsize=14)
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0, frameon=False)


ax = plt.subplot(2,2,3)
ax.set_xlabel('time (s)', fontsize=14)
ln1 = ax.plot(time_mag, zefit,'k', linewidth=2,label = 'Z EFIT')
ax.set_ylabel('Z (m)', fontsize=14)
ax2=ax.twinx()
ln2 = ax2.plot(t, asimhf, 'r', linewidth=2,label = 'Asym')
ax2.set_ylabel('Asymmetry', fontsize=14)
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0, frameon=False)


ax = plt.subplot(2,2,4)
ax.set_xlabel('time (s)', fontsize=14)
ln1 = ax.plot(time_mag, refit,'k', linewidth=2,label = 'R EFIT')
ax.set_ylabel('R (m)', fontsize=14)
ax2=ax.twinx()
ln2 = ax2.plot(t, asimvf, 'r', linewidth=2,label = 'Asym')
ax2.set_ylabel('Asymmetry', fontsize=14)
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0, frameon=False)

plt.savefig(str(shot) + '.png')


# In this part I plot the figure which put in relation the asymmetry with R and Z
refit_new = np.interp(t, time_mag, refit)
zefit_new = np.interp(t, time_mag, zefit)


plt.figure(figsize = (23, 17))
plt.subplot(1,2,1)
plt.scatter(zefit_new, asimhf)
plt.xlabel('Z (m)', fontsize=14)
plt.ylabel('Asymmetry', fontsize=14)

plt.subplot(1,2,2)
plt.scatter(refit_new, asimvf)
plt.xlabel('R (m)', fontsize=14)
plt.ylabel('Asymmetry', fontsize=14)







