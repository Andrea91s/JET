# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:04:10 2020

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib import cm
from scipy.stats import kde


def interpolation(newtime, oldtime, array):
    return np.interp(newtime, oldtime, array)
def f(x,a,b):
    return a*x +b



#f= open("EFIT.txt","w+")
#shot=np.linspace(82307,97781,15475, dtype = int)
#
#for w in range(0, len(shot)):    
#    try:
#        refit = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/REFIT_'+ str(shot[w])+'.txt')
#        print(shot[w])
#        f.write(str(shot[w]) + ' \n')    	
#    except:
#        print(shot[w], 'does not exist in the database')
#        f.write("does not exist in the database\n") 
#f.close()

#zefit_big = []
#refit_big = []
#zeftp_big = []
#reftp_big = []
#
#
#goodpulses = np.loadtxt('EFIT.txt')
#for i in range(len(goodpulses)):
#    print(goodpulses[i])
#    test = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCtime_'+ str(int(goodpulses[0]))+'.txt')
#    refit = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/REFIT_'+ str(int(goodpulses[0]))+'.txt')
#    zefit = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ZEFIT_'+ str(int(goodpulses[0]))+'.txt')
#    tefit = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/time_mag_'+ str(int(goodpulses[0]))+'.txt')
#
#    rftp = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/python/R/REFTP_'+ str(int(goodpulses[0]))+'.txt')
#    zftp = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/python/Z/ZEFTP_'+ str(int(goodpulses[0]))+'.txt')
#    tftp = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/python/T/time_mag_'+ str(int(goodpulses[0]))+'.txt')
#
#
#    new_refit = interpolation(tftp,tefit, refit)
#    new_zefit = interpolation(tftp,tefit, zefit)
#
#    
#    zefit_big = np.append(zefit_big,new_zefit)
#    refit_big = np.append(refit_big,new_refit)
#    zeftp_big = np.append(zeftp_big,zftp)
#    reftp_big = np.append(reftp_big,rftp)
    
#np.savetxt('zefit_big.dat',zefit_big)
#np.savetxt('refit_big.dat',refit_big)
#np.savetxt('zeftp_big.dat',zeftp_big)
#np.savetxt('reftp_big.dat',reftp_big)
    
#zefit_big=np.loadtxt('zefit_big.dat')
#refit_big=np.loadtxt('refit_big.dat')
#zeftp_big=np.loadtxt('zeftp_big.dat')
#reftp_big=np.loadtxt('reftp_big.dat')

idx = np.where((refit_big<3.04) & (refit_big>2.95))
refit_biga = refit_big[idx]
reftp_biga = reftp_big[idx]

idx = np.where((zefit_big>0.22) & (zefit_big<0.32))
zefit_biga = zefit_big[idx]
zeftp_biga = zeftp_big[idx]
#
#is_above = lambda p,a,b: np.cross(p-a, b-a) < 0
#a = np.array([0,-0.327143]) # [x,y]
#b = np.array([4,4.088108]) # [x,y]
#above=[]
#below = []
#for i in range(len(reftp_biga)):
#    print(i)
#    if is_above([refit_biga[i], reftp_biga[i]],a,b):
#        above=np.append(above,i)
#    else:
#        below=np.append(below,i)
#
#


reftp_biga[above.astype(int)] = reftp_biga[above.astype(int)]-0.0055
reftp_biga[below.astype(int)] = reftp_biga[below.astype(int)]+0.0085


fitz = LinearRegression().fit(zefit_biga.reshape((-1, 1)), zeftp_biga)
fitr = LinearRegression().fit(refit_biga.reshape((-1, 1)), reftp_biga)


r_z = fitz.score(zefit_biga.reshape((-1, 1)), zeftp_biga)
r_r = fitr.score(refit_biga.reshape((-1, 1)), reftp_biga)

#zefit_big_test = []
#refit_big_test = []
#zeftp_big_test = []
#reftp_big_test = []
#
testpulses = [94270,94665,82812,96435]
for i in range(len(testpulses)):
    print(testpulses[i])
    
    refit = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/REFIT_'+ str(int(testpulses[i]))+'.txt')
    zefit = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ZEFIT_'+ str(int(testpulses[i]))+'.txt')
    tefit = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/time_mag_'+ str(int(testpulses[i]))+'.txt')

    rftp = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/python/R/REFTP_'+ str(int(testpulses[i]))+'.txt')
    zftp = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/python/Z/ZEFTP_'+ str(int(testpulses[i]))+'.txt')
    tftp = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/python/T/time_mag_'+ str(int(testpulses[i]))+'.txt')


    new_refit_test = interpolation(tftp,tefit, refit)
    new_zefit_test = interpolation(tftp,tefit, zefit)
#
#    
#    zefit_big_test = np.append(zefit_big_test,new_zefit_test)
#    refit_big_test = np.append(refit_big_test,new_refit_test)
#    zeftp_big_test = np.append(zeftp_big_test,zftp)
#    reftp_big_test = np.append(reftp_big_test,rftp)

plt.close('all')
plt.figure(1,figsize = (11, 6))
plt.subplot(1,2,1)
plt.scatter(zefit_biga, zeftp_biga,c='r', s=2)
#plt.scatter(zefit_big_test, zeftp_big_test,c='b', s=4)
plt.plot(np.linspace(min(zefit_biga), max(zefit_biga), 100), 
         np.linspace(min(zefit_biga), max(zefit_biga), 100)*fitz.coef_ + fitz.intercept_, 'b')
plt.xlabel('EFIT Z (m)')
plt.ylabel('EFTP Z (m)')
plt.xlim(0.20, 0.34)
plt.ylim(0.20, 0.34)

#plt.close('all')
#
#plt.figure(2)
plt.subplot(1,2,2)

plt.scatter(refit_biga, reftp_biga,c='r', s=2, marker='+')
#plt.scatter(refit_big_test, reftp_big_test,c='b', s=4)
plt.plot(np.linspace(min(refit_biga), max(refit_biga), 100), 
         np.linspace(min(refit_biga), max(refit_biga), 100)*fitr.coef_ + fitr.intercept_, 'b')
plt.xlabel('EFIT R (m)')
plt.ylabel('EFTP R (m)')
plt.xlim(2.90, 3.05)
plt.ylim(2.90, 3.05)







#plt.plot(np.linspace(min(refit_biga), max(refit_biga), 100), np.linspace(min(refit_biga), max(refit_biga), 100)*fitr.coef_ + fitr.intercept_, 'b')
#plt.xlabel('EFIT R (m)')
#plt.ylabel('EFTP R (m)')



#def niceplot(x,y):
#    data = np.zeros((len(x),2))
#    data[:,0]=x
##    print(min(x))
##    print(max(x))
##    print(min(y))
##    print(max(y))
#    data[:,1]=y
#    k = kde.gaussian_kde(data.T)
#    nbins=100
#    xi,yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]  
#    zi= k(np.vstack([xi.flatten(), yi.flatten()]))
#    plt.figure()
#    plt.pcolormesh(xi,yi,zi.reshape(xi.shape),cmap=cm.hot_r)
#    plt.colorbar()
#    #fig=zi.reshape(xi.shape)
#    #return xi,yi,zi.reshape(xi.shape)
##
#niceplot(refit_biga,reftp_biga)



