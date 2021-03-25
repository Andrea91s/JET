#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 18:15:49 2020

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression
from matplotlib import gridspec
from matplotlib import cm
import random as rnd
from scipy.stats import kde
import seaborn as snb
#plt.close('all')

def niceplot(x,y,a):
    data = np.zeros((len(x),2))
    data[:,0]=x
    print(min(x))
    print(max(x))
    print(min(y))
    print(max(y))
    data[:,1]=y
    k = kde.gaussian_kde(data.T)
    nbins=250
    xi,yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]  
    zi= k(np.vstack([xi.flatten(), yi.flatten()]))
    #plt.figure()
    #plt.pcolormesh(xi,yi,zi.reshape(xi.shape),cmap=cm.hot_r)
    fig=zi.reshape(xi.shape)
    if   a==1:
        np.savetxt('/home/andrea/PAPERS/JETREALTIME/Figure2/Za_new.dat',fig) 
    elif a==2:
        np.savetxt('/home/andrea/PAPERS/JETREALTIME/Figure2/Ra_new.dat',fig) 
    elif a==3:
        np.savetxt('/home/andrea/PAPERS/JETREALTIME/Figure2/Zm_new.dat',fig)
    elif a==4:
        np.savetxt('/home/andrea/PAPERS/JETREALTIME/Figure2/Rm_new.dat',fig)     
#    plt.contour(xi,yi,zi.reshape(xi.shape),'k')
    return xi,yi,zi.reshape(xi.shape)



hwei=np.loadtxt('PAPERZM_new')

vwei=np.loadtxt('PAPERRM_new')

zefit=np.loadtxt('PAPERZEFITA_new')

refit=np.loadtxt('PAPERREFITA_new')

zefitm=np.loadtxt('PAPERZEFITM_new')

refitm=np.loadtxt('PAPERREFITM_new')

hasim=np.loadtxt('PAPERZA_new')

vasim=np.loadtxt('PAPERRA_new')


#niceplot(zefit,hasim,a=1)
#niceplot(refit,vasim,a=2)
#niceplot(zefitm,hwei,a=3)
#niceplot(refitm,vwei,a=4)

#
#fitaH = LinearRegression().fit(zefit.reshape((-1, 1)), hasim)
#fitaV = LinearRegression().fit(refit.reshape((-1, 1)), vasim)
#fitmH = LinearRegression().fit(zefitm.reshape((-1, 1)), hwei)
#fitmV = LinearRegression().fit(refitm.reshape((-1, 1)), vwei)
#
#    
#print('The a coefficient for the asymmetry HC is = ', round(fitaH.intercept_,5))
#print('The b coefficient for the asymmetry HC is = ',  round(fitaH.coef_[0],5))
#
#print('The a coefficient for the asymmetry VC is = ',  round(fitaV.intercept_,5))
#print('The b coefficient for the asymmetry VC is = ',    round(fitaV.coef_[0],5))
#print('')
#print('The a coefficient for the weight HC is = ', round(fitmH.intercept_,5))
#print('The b coefficient for the weight HC is = ',  round(fitmH.coef_[0],5))
#
#print('The a coefficient for the weight  VC is = ',  round(fitmV.intercept_,5))
#print('The b coefficient for the weight  VC is = ',    round(fitmV.coef_[0],5))
#
#
#r_sqaH = fitaH.score(zefit.reshape((-1, 1)),hasim)
#r_sqaV = fitaV.score(refit.reshape((-1, 1)),vasim)
#r_sqmH = fitmH.score(zefitm.reshape((-1, 1)),hwei)
#r_sqmV = fitmV.score(refitm.reshape((-1, 1)),vwei)
#
# 
#plt.figure(400,figsize = (24, 17))
#plt.subplot(2,2,1)    
#plt.plot(Zm, fitmH.intercept_+Zm*fitmH.coef_, c='r', linewidth=3, zorder=10)
##plt.plot(Zm, 5.5-4.2*Zm, c='g', linewidth=3, zorder=10)
#plt.text(np.max(zefitm), np.min(hwei), '$R^2$ = (%.3f)'%r_sqmH,
#        verticalalignment='bottom', horizontalalignment='right', fontsize=25) 
#plt.scatter(zefitm, hwei, c='b', s=10)
#plt.xlabel('Z (m)', fontsize=20)
#plt.ylabel('WEIGHT H', fontsize=20)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#
#
#plt.subplot(2,2,2)
#plt.scatter(refitm, vwei, c='b', s=10)
#plt.plot(Rm, fitmV.intercept_+Rm*fitmV.coef_, c='r', linewidth=3, zorder=10)
##plt.plot(Rm, 4.8+Rm*3.1, c='g', linewidth=3, zorder=10)
#plt.text(np.max(refitm), np.min(vwei), '$R^2$ = (%.3f)'%r_sqmV,
#        verticalalignment='bottom', horizontalalignment='right', fontsize=25) 
#plt.xlabel('R (m)', fontsize=20)
#plt.ylabel('WEIGHT V', fontsize=20)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#
#
#plt.subplot(2,2,3)
#plt.plot(Z, fitaH.intercept_+Z*fitaH.coef_, c='r', linewidth=3, zorder=10)
##plt.plot(Z, -0.56+Z*2.3, c='g', linewidth=3, zorder=10)
#plt.text(np.max(zefit), np.min(hasim), '$R^2$ = (%.3f)'%r_sqaH,
#        verticalalignment='bottom', horizontalalignment='right', fontsize=25) 
#plt.scatter(zefit, hasim, c='b', s=10)
#plt.xlabel('Z (m)', fontsize=20)
#plt.ylabel('Asymmetry H', fontsize=20)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#
#
#
#plt.subplot(2,2,4)
#plt.plot(R, fitaV.intercept_+R*fitaV.coef_, c='r', linewidth=3, zorder=10)
##plt.plot(R, 4.9+R*-1.6, c='g', linewidth=3, zorder=10)
##    plt.plot(R, p(R), c='r', linewidth=3, zorder=10)
#plt.text(np.max(refit), np.max(vasim), '$R^2$ = (%.3f)'%r_sqaV,
#         verticalalignment='top', horizontalalignment='right', fontsize=25) 
#plt.scatter(refit, vasim,c='b',s=10)
#plt.xlabel('R (m)', fontsize=20)
#plt.ylabel('Asymmetry V', fontsize=20)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#
#plt.savefig('zzz' + str(scenario[w]) + 'fit.png')
#
#plt.close('all')
    

