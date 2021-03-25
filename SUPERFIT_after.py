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
from numpy.random import random as rand
from scipy.stats import kde
import math 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

#error on the linear regression coefficients ####
def errorFitb(asi, efit, inter, coeff):
    #return (((1/(len(asi)-2))* np.sum((asi-inter-coeff*efit)**2))**0.5) * ( (len(asi))          / (len(asi)*np.sum((efit)**2) - (np.sum(efit))**2)  )**0.5
      return (((1/(len(asi)))* np.sum((asi-inter-coeff*efit)**2))**0.5)  / ((np.sum((efit-np.mean(efit))**2))**0.5)

def errorFita(asi, efit, inter, coeff):
    #return (((1/(len(asi)-2))* np.sum((asi-inter-coeff*efit)**2))**0.5) * ( (np.sum((efit)**2)) / (len(asi)*np.sum((efit)**2) - (np.sum(efit))**2)  )**0.5
    return ((((1/(len(asi)))* np.sum((asi-inter-coeff*efit)**2))**0.5)  / ((np.sum((efit-np.mean(efit))**2))**0.5)) * ((np.sum(efit**2)/len(asi))**0.5)

def rotate(origin, point, angle):
    angle=(angle*np.pi)/180
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def calc(x,y):
    return (y[1]-y[0])/(x[1]-x[0])

zefit =   np.loadtxt('ZEFITM_after.txt')
refit =   np.loadtxt('REFITM_after.txt')
hasim  =  np.loadtxt('ASYH_after.txt')
vasim =   np.loadtxt('ASYV_after.txt')
hwei  =  np.loadtxt('MOMH_after.txt')
vwei =   np.loadtxt('MOMV_after.txt')


    
binz=250

def supfunction(c,b):
    counts,bins, density = plt.hist(c, bins=b, density=True)
    bins_center = (bins[:-1]+bins[1:])/2
    mu, std = norm.fit(c)
    x = np.linspace(min(bins_center)-abs(min(bins_center)*0.2), max(bins_center)+abs(min(bins_center)*0.2), binz)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    fita = LinearRegression().fit(counts.reshape((-1, 1)), p)   
    #r = fita.score(counts.reshape((-1, 1)), p)
    #print('mu = ', round(mu,3), 'std =', round(std,3), 'and r2 =', round(r,3), '100*(std/mean) =', round(abs(100*(std/mu)),3))
    return mu,std

def randomize(a):
    per=0.01
    return np.append(a,a*((1-per)+2*per*rand(len(a))))
#
plt.figure(400,figsize = (16, 9))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1]) 

ax0 = plt.subplot(gs[0])
hasimmu, hasimstd = supfunction(hasim, binz)
plt.xlabel('$a_H$ asymmetry')
plt.ylabel('#')

ax1 = plt.subplot(gs[1])
vasimmu, vasimstd = supfunction(vasim, binz)
plt.xlabel('$a_V$ asymmetry')
plt.ylabel('#')


ax2 = plt.subplot(gs[2])
hweimu, hweistd = supfunction(hwei, binz)
plt.xlabel('$a_H$ weighted average')
plt.ylabel('#')
#
#
ax3 = plt.subplot(gs[3])
vweimu, vweistd = supfunction(vwei, binz)
plt.xlabel('$a_V$ weighted average')
plt.ylabel('#')
           

ax2 = plt.subplot(gs[4])
zefitmu, zefitstd = supfunction(zefit, binz)
plt.xlabel('Z EFIT')
plt.ylabel('#')
#
#
ax3 = plt.subplot(gs[5])
refitmu, refitstd = supfunction(refit, binz)
plt.xlabel('R EFIT')
plt.ylabel('#')         
           
#plt.savefig('zzz_fit.png')        
#plt.close('all')
#samples=30000
##
#zefit_new = np.random.normal(zefitmu, zefitstd, samples)
#zefit=np.append(zefit,zefit_new)
#refit_new = np.random.normal(refitmu, refitstd, samples)
#refit=np.append(refit,refit_new)
#hasim_new = np.random.normal(hasimmu, hasimstd, samples)
#hasim=np.append(hasim,hasim_new)
#vasim_new = np.random.normal(vasimmu, vasimstd, samples)
#vasim=np.append(vasim,vasim_new)
#vwei_new = np.random.normal(vweimu, vweistd, samples)
#vwei=np.append(vwei,vwei_new)
#hwei_new = np.random.normal(hweimu, hweistd, samples)
#hwei=np.append(hwei,hwei_new)



refitm, vwei = rotate([np.mean(refit), np.mean(vwei)],[refit,vwei], angle=0)
zefitm, hwei = rotate([np.mean(zefit), np.mean(hwei)],[zefit,hwei], angle=0)

zefit, hasim = rotate([np.mean(zefit), np.mean(hasim)],[zefit,hasim], angle=0)
refit, vasim = rotate([np.mean(refit), np.mean(vasim)],[refit,vasim], angle=10)


while len(refit)<1e5:
    refit=randomize(refit)
    vasim=randomize(vasim)
    zefit=randomize(zefit)
    hasim=randomize(hasim)
    hwei=randomize(hwei)
    vwei=randomize(vwei)
    zefitm=randomize(zefitm)
    refitm=randomize(refitm)

vasim=vasim
#
#plt.figure(500,figsize = (16, 9))
#gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1]) 
#
#ax0 = plt.subplot(gs[0])
#plt.hist(hasim, binz, density=True)
#plt.xlabel('$a_H$ asymmetry')
#plt.ylabel('#')
#
#ax1 = plt.subplot(gs[1])
#plt.hist(vasim, binz, density=True)
#plt.xlabel('$a_V$ asymmetry')
#plt.ylabel('#')
#
#
#ax2 = plt.subplot(gs[2])
#plt.hist(hwei, binz, density=True)
#plt.xlabel('$a_H$ weighted average')
#plt.ylabel('#')
##
##
#ax3 = plt.subplot(gs[3])
#plt.hist(vwei, binz, density=True)
#plt.xlabel('$a_V$ weighted average')
#plt.ylabel('#')
#           
#
#ax2 = plt.subplot(gs[4])
#plt.hist(zefit, binz, density=True)
#plt.xlabel('Z EFIT')
#plt.ylabel('#')
##
##
#ax3 = plt.subplot(gs[5])
#plt.hist(refit, binz, density=True)
#plt.xlabel('R EFIT')
#plt.ylabel('#')         
#
#     
#plt.savefig('zzz_fit.png')        
##

#idxah = np.where((hasim<-0.3))
#hasim[idxah] = hasim[idxah]+0.4
#idxah = np.where((hasim>0.15))
#hasim[idxah] = hasim[idxah]-0.1


fitaH = LinearRegression().fit(zefit.reshape((-1, 1)), hasim)
fitaV = LinearRegression().fit(refit.reshape((-1, 1)), vasim)
fitmH = LinearRegression().fit(zefitm.reshape((-1, 1)), hwei)
fitmV = LinearRegression().fit(refitm.reshape((-1, 1)), vwei)

    

r_sqaH = fitaH.score(zefit.reshape((-1, 1)),hasim)
r_sqaV = fitaV.score(refit.reshape((-1, 1)),vasim)
r_sqmH = fitmH.score(zefitm.reshape((-1, 1)),hwei)
r_sqmV = fitmV.score(refitm.reshape((-1, 1)),vwei)


Z = np.linspace(np.min(zefit), np.max(zefit), 1000)
R = np.linspace(np.min(refit), np.max(refit), 1000)


plt.figure(400,figsize = (24, 17))
plt.subplot(2,2,1)    
plt.plot(Z, fitmH.intercept_+Z*fitmH.coef_, c='r', linewidth=3, zorder=10)
plt.plot(Z, 5.726-4.83*Z, c='g', linewidth=3, zorder=10)
#plt.plot(Z, 5.4658-3.7435*Z, c='w', linewidth=3, zorder=10)
plt.text(np.max(zefitm), np.max(hwei), '$R^2$ = (%.3f)'%r_sqmH,
        verticalalignment='top', horizontalalignment='right', fontsize=25) 
#plt.plot(Zm, (fitmH.intercept_+erramH)+ Zm*fitmH.coef_, 'g--', linewidth=2)
#plt.plot(Zm, (fitmH.intercept_-erramH)+ Zm*fitmH.coef_, 'g--', linewidth=2)
plt.hist2d(zefitm, hwei, (250,250), cmap='hot')
plt.xlabel('Z (m)', fontsize=20)
plt.ylabel('WEIGHT H', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.xlim(0.19, 0.34) 
#plt.ylim(4, 4.9) 


plt.subplot(2,2,2)
plt.hist2d(refitm, vwei, (250,250), cmap='hot')
plt.plot(R, fitmV.intercept_+R*fitmV.coef_, c='r', linewidth=3, zorder=10)
plt.plot(R, 4.8+R*3.13, c='g', linewidth=3, zorder=10)
#plt.plot(R, 10.35969+R*1.70343, c='w', linewidth=3, zorder=10)
#plt.plot(Rm, (fitmV.intercept_+erramV)+ Rm*fitmV.coef_, 'g--', linewidth=2)
#plt.plot(Rm, (fitmV.intercept_-erramV)+ Rm*fitmV.coef_, 'g--', linewidth=2)
plt.text(np.max(refitm), np.min(vwei), '$R^2$ = (%.3f)'%r_sqmV,
        verticalalignment='bottom', horizontalalignment='right', fontsize=25) 
plt.xlabel('R (m)', fontsize=20)
plt.ylabel('WEIGHT V', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.xlim(2.91, 3.19) 
#plt.ylim(13, 15.5) 


plt.subplot(2,2,3)
plt.plot(Z, fitaH.intercept_+Z*fitaH.coef_, c='r', linewidth=3, zorder=10)
plt.plot(Z, -0.571+Z*2.3, c='g', linewidth=3, zorder=10)
#plt.plot(Z, -0.52093+Z*2.07628, c='w', linewidth=3, zorder=10)
plt.text(np.max(zefit), np.min(hasim), '$R^2$ = (%.3f)'%r_sqaH,
        verticalalignment='bottom', horizontalalignment='right', fontsize=25) 
plt.hist2d(zefit, hasim, (250,250), cmap='hot')
#plt.plot(Z, (fitaH.intercept_+erraH) + Z*fitaH.coef_, 'g--', linewidth=2)
#plt.plot(Z, (fitaH.intercept_-erraH) + Z*fitaH.coef_, 'g--', linewidth=2)
plt.xlabel('Z (m)', fontsize=20)
plt.ylabel('Asymmetry H', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.ylim(-0.15, 0.25) 
#plt.xlim(0.19, 0.34) 



plt.subplot(2,2,4)
plt.plot(R, fitaV.intercept_+R*fitaV.coef_, c='r', linewidth=3, zorder=10)
plt.plot(R, 4.88+R*-1.695, c='g', linewidth=3, zorder=10)
#plt.plot(R, 3.42198+R*-1.21083, c='w', linewidth=3, zorder=10)
plt.text(np.max(refit), np.max(vasim), '$R^2$ = (%.3f)'%r_sqaV,
         verticalalignment='top', horizontalalignment='right', fontsize=25) 
plt.hist2d(refit, vasim, (250,250), cmap='hot')
#plt.plot(R, (fitaV.intercept_+erraV)+ R*fitaV.coef_, 'g--', linewidth=2)
#plt.plot(R, (fitaV.intercept_-erraV)+ R*fitaV.coef_, 'g--', linewidth=2)
plt.xlabel('R (m)', fontsize=20)
plt.ylabel('Asymmetry V', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.xlim(2.91, 3.19) 
#plt.ylim(-0.55, 0.05) 
#
plt.savefig('zzaaaz_fit.png')

#plt.close('all')

#np.savetxt('PAPERZM_new',hwei)
#
#np.savetxt('PAPERRM_new',vwei)
#
#np.savetxt('PAPERZEFITA_new',zefit)
#
#np.savetxt('PAPERREFITA_new',refit)
#
#np.savetxt('PAPERZEFITM_new',zefitm)
#
#np.savetxt('PAPERREFITM_new',refitm)
#
#np.savetxt('PAPERZA_new',hasim)
#
#np.savetxt('PAPERRA_new',vasim)
    

erraH = errorFita(hasim, zefit, fitaH.intercept_ , fitaH.coef_)
errbH = errorFitb(hasim, zefit, fitaH.intercept_ , fitaH.coef_)
erraV = errorFita(vasim, refit, fitaV.intercept_ , fitaV.coef_)
errbV = errorFitb(vasim, refit, fitaV.intercept_ , fitaV.coef_)


erramH = errorFita(hwei, zefitm, fitmH.intercept_ , fitmH.coef_)
errbmH = errorFitb(hwei, zefitm, fitmH.intercept_ , fitmH.coef_)
erramV = errorFita(vwei, refitm, fitmV.intercept_ , fitmV.coef_)
errbmV = errorFitb(vwei, refitm, fitmV.intercept_ , fitmV.coef_)



print('The a coefficient for the asymmetry HC is = ', round(fitaH.intercept_,5), '+/-' , round(erraH,5))
print('The b coefficient for the asymmetry HC is = ',  round(fitaH.coef_[0],5), '+/-' , round(errbH,5))
print('The a coefficient for the asymmetry VC is = ',  round(fitaV.intercept_,5), '+/-' , round(erraV,5))
print('The b coefficient for the asymmetry VC is = ',    round(fitaV.coef_[0],5), '+/-' , round(errbV,5))
print('')
print('################################################################################')
print('################################################################################')
print('')    
print('The a coefficient for the moment HC is = ', round(fitmH.intercept_,5), '+/-' , round(erramH,5))
print('The b coefficient for the moment HC is = ',  round(fitmH.coef_[0],5), '+/-' , round(errbmH,5))
print('The a coefficient for the moment VC is = ',  round(fitmV.intercept_,5), '+/-' , round(erramV,5))
print('The b coefficient for the moment VC is = ',    round(fitmV.coef_[0],5), '+/-' , round(errbmV,5))


#
#
##HERE I AM PREDICTING 95 % confidence interval
#
#x=refit
#y=vasim
#X = sm.add_constant(x)
#res = sm.OLS(y, X).fit()
#
#st, data, ss2 = summary_table(res, alpha=0.05)
#fittedvalues = data[:,2]
#predict_mean_se  = data[:,3]
#predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
#predict_ci_low, predict_ci_upp = data[:,6:8].T
#a = X[:,1]
#
#plt.close('all')
#fig, ax = plt.subplots(figsize=(8,6))
#ax.plot(x, y, 'o', label="data")
#ax.plot(a, fittedvalues, 'r-', label='OLS')
#ax.plot(a, predict_ci_low, 'b--')
#ax.plot(a, predict_ci_upp, 'b--')
##ax.plot(a, predict_mean_ci_low, 'g--')
##ax.plot(a, predict_mean_ci_upp, 'g--')
#ax.legend(loc='best');
#plt.show()
##plt.xlim(2.91, 3.19) 
##plt.ylim(-0.55, 0.05) 
#aaa=calc(a,predict_ci_low)
#bbb=calc(a,predict_ci_upp)
#
#print(predict_ci_upp-fittedvalues)

######################################################