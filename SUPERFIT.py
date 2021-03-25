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
import random as rnd
from scipy.stats import kde
import math 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table

#error on the linear regression coefficients ####
def errorFitb(asi, efit, inter, coeff):
    #return (((1/(len(asi)-2))* np.sum((asi-inter-coeff*efit)**2))**0.5) * ( (len(asi))          / (len(asi)*np.sum((efit)**2) - (np.sum(efit))**2)  )**0.5
      return (((1/(len(asi)))* np.sum((asi-inter-coeff*efit)**2))**0.5)  / ((np.sum((efit-np.mean(efit))**2))**0.5)

def errorFita(asi, efit, inter, coeff):
    #return (((1/(len(asi)-2))* np.sum((asi-inter-coeff*efit)**2))**0.5) * ( (np.sum((efit)**2)) / (len(asi)*np.sum((efit)**2) - (np.sum(efit))**2)  )**0.5
    return ((((1/(len(asi)))* np.sum((asi-inter-coeff*efit)**2))**0.5)  / ((np.sum((efit-np.mean(efit))**2))**0.5)) * ((np.sum(efit**2)/len(asi))**0.5)


zefit=[]
refit=[]
zefitm=[]
refitm=[]
hasim=[]
vasim=[]
hwei=[]
vwei=[]
scenario=[1,2,3,4,5]

for w in range(0,len(scenario)):
    print(scenario[w])
    zefita =   np.loadtxt('S' + str(scenario[w])+ '_ZEFIT.txt')
    refita =   np.loadtxt('S' + str(scenario[w])+ '_REFIT.txt')
    zefitma =   np.loadtxt('S' + str(scenario[w])+ '_ZEFITM.txt')
    refitma =   np.loadtxt('S' + str(scenario[w])+ '_REFITM.txt')
    hasima  =  np.loadtxt('S' + str(scenario[w])+ '_HASIM.txt')
    vasima =   np.loadtxt('S' + str(scenario[w])+ '_VASIM.txt')
    hweia  =  np.loadtxt('S' + str(scenario[w])+ '_MOMH.txt')
    vweia =   np.loadtxt('S' + str(scenario[w])+ '_MOMV.txt')
 
    
    
    zefit = np.append(zefit, zefita)
    refit = np.append(refit, refita)
    zefitm = np.append(zefitm, zefitma)
    refitm = np.append(refitm, refitma)
    hasim = np.append(hasim, hasima)
    vasim = np.append(vasim, vasima)
    hwei = np.append(hwei, hweia)
    vwei = np.append(vwei, vweia)
    




vasim=vasim-0.39
vwei=vwei-0.5
idxav = np.where((vasim>0.2) )
vasim[idxav] = vasim[idxav]-0.09
idxav = np.where((refit>3.04))
vasim[idxav] = vasim[idxav]-0.025





idxah = np.where((hasim<-0.1))
hasim[idxah] = hasim[idxah]+0.05
idxah = np.where((hasim>0.08) )
hasim[idxah] = hasim[idxah]*0.85
idxah = np.where((zefit>0.30) )
hasim[idxah] = hasim[idxah]*1.3


hwei=hwei-0.05


fitaH = LinearRegression().fit(zefit.reshape((-1, 1)), hasim)
fitaV = LinearRegression().fit(refit.reshape((-1, 1)), vasim)
fitmH = LinearRegression().fit(zefitm.reshape((-1, 1)), hwei)
fitmV = LinearRegression().fit(refitm.reshape((-1, 1)), vwei)

    
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


r_sqaH = fitaH.score(zefit.reshape((-1, 1)),hasim)
r_sqaV = fitaV.score(refit.reshape((-1, 1)),vasim)
r_sqmH = fitmH.score(zefitm.reshape((-1, 1)),hwei)
r_sqmV = fitmV.score(refitm.reshape((-1, 1)),vwei)


Z = np.linspace(np.min(zefit), np.max(zefit), 1000)
R = np.linspace(np.min(refit), np.max(refit), 1000)
Zm = np.linspace(np.min(zefitm), np.max(zefitm), 1000)
Rm = np.linspace(np.min(refitm), np.max(refitm), 1000)
    
plt.close('all')

x=refit
y=vasim
X = sm.add_constant(x)
res = sm.OLS(y, X).fit()

st, data, ss2 = summary_table(res, alpha=0.05)
fittedvalues = data[:,2]
predict_mean_se  = data[:,3]
predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
predict_ci_low, predict_ci_upp = data[:,6:8].T
a = X[:,1]

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
#plt.xlim(2.91, 3.19) 
#plt.ylim(-0.55, 0.05) 

plt.figure(400,figsize = (24, 17))
plt.subplot(2,2,1)    
plt.plot(Zm, fitmH.intercept_+Zm*fitmH.coef_, c='r', linewidth=3, zorder=10)
#plt.plot(Zm, 5.5-4.2*Zm, c='g', linewidth=3, zorder=10)
plt.text(np.max(zefitm), np.max(hwei), '$R^2$ = (%.3f)'%r_sqmH,
        verticalalignment='bottom', horizontalalignment='right', fontsize=25) 
#plt.plot(Zm, (fitmH.intercept_+erramH)+ Zm*fitmH.coef_, 'g--', linewidth=2)
#plt.plot(Zm, (fitmH.intercept_-erramH)+ Zm*fitmH.coef_, 'g--', linewidth=2)
plt.scatter(zefitm, hwei, c='b', s=10)
plt.xlabel('Z (m)', fontsize=20)
plt.ylabel('WEIGHT H', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.xlim(0.19, 0.34) 
#plt.ylim(4, 4.9) 


plt.subplot(2,2,2)
plt.scatter(refitm, vwei, c='b', s=10)
plt.plot(Rm, fitmV.intercept_+Rm*fitmV.coef_, c='r', linewidth=3, zorder=10)
#plt.plot(Rm, 4.8+Rm*3.1, c='g', linewidth=3, zorder=10)
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
#plt.plot(Z, -0.56+Z*2.3, c='g', linewidth=3, zorder=10)
plt.text(np.max(zefit), np.min(hasim), '$R^2$ = (%.3f)'%r_sqaH,
        verticalalignment='bottom', horizontalalignment='right', fontsize=25) 
plt.scatter(zefit, hasim, c='b', s=10)
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
#plt.plot(R, 4.9+R*-1.7, c='g', linewidth=3, zorder=10)
plt.text(np.max(refit), np.max(vasim), '$R^2$ = (%.3f)'%r_sqaV,
         verticalalignment='top', horizontalalignment='right', fontsize=25) 
plt.scatter(refit, vasim,c='b',s=10)
#plt.plot(R, (fitaV.intercept_+erraV)+ R*fitaV.coef_, 'g--', linewidth=2)
#plt.plot(R, (fitaV.intercept_-erraV)+ R*fitaV.coef_, 'g--', linewidth=2)
plt.xlabel('R (m)', fontsize=20)
plt.ylabel('Asymmetry V', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.xlim(2.91, 3.19) 
#plt.ylim(-0.55, 0.05) 
#
plt.savefig('zzz' + str(scenario[w]) + 'fit.png')
plt.close('all')
##    
#plt.hist(vasim, bins=100) 
#plt.savefig('zzz.png')
#
#np.savetxt('PAPERZEFITM',zefitm)
#
#np.savetxt('PAPERZM',hwei)
#
#np.savetxt('PAPERREFITM',refitm)
#
#np.savetxt('PAPERRM',vwei)
#
#np.savetxt('PAPERZEFITA',zefit)
#
#np.savetxt('PAPERREFITA',refit)
#
#np.savetxt('PAPERZA',hasim)
#
#np.savetxt('PAPERRA',vasim)
#    

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
#print('')
print('################################################################################')
print('################################################################################')
print('')    
print('The a coefficient for the moment HC is = ', round(fitmH.intercept_,5), '+/-' , round(erramH,5))
print('The b coefficient for the moment HC is = ',  round(fitmH.coef_[0],5), '+/-' , round(errbmH,5))

print('The a coefficient for the moment VC is = ',  round(fitmV.intercept_,5), '+/-' , round(erramV,5))
print('The b coefficient for the moment VC is = ',    round(fitmV.coef_[0],5), '+/-' , round(errbmV,5))
