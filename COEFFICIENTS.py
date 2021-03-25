#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:59:19 2020

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.utils import shuffle
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import scipy.stats as stats


shota = np.loadtxt('COEFF_SHOT.txt')
coeffaha = np.loadtxt('COEFF_ASYM_H_A.txt')
coeffahb = np.loadtxt('COEFF_ASYM_H_B.txt')
coeffwha = np.loadtxt('COEFF_WEIG_H_A.txt')
coeffwhb = np.loadtxt('COEFF_WEIG_H_B.txt')
coeffava = np.loadtxt('COEFF_ASYM_V_A.txt')
coeffavb = np.loadtxt('COEFF_ASYM_V_B.txt')
coeffwva = np.loadtxt('COEFF_WEIG_V_A.txt')
coeffwvb = np.loadtxt('COEFF_WEIG_V_B.txt')



newlength = 2000
shot = np.linspace(0,len(shota), len(shota))
shotnew = np.linspace(0,len(shota), newlength)

coeffaha = shuffle(np.interp(shotnew, shot, coeffaha))
coeffahb = shuffle(np.interp(shotnew, shot, coeffahb))
coeffava = shuffle(np.interp(shotnew, shot, coeffava))
coeffavb = shuffle(np.interp(shotnew, shot, coeffavb))

coeffwha = shuffle(np.interp(shotnew, shot, coeffwha))
coeffwhb = shuffle(np.interp(shotnew, shot, coeffwhb))
coeffwva = shuffle(np.interp(shotnew, shot, coeffwva))
coeffwvb = shuffle(np.interp(shotnew, shot, coeffwvb))

#print('The mean value for horizontal asymmetry a is', round(np.mean(coeffaha),4), '+-' ,round(np.std(coeffaha),4), 'and the error % is', abs(round(100*(np.std(coeffaha)/np.mean(coeffaha)),3)),'%')
#print('The mean value for horizontal asymmetry b is', round(np.mean(coeffahb),4), '+-' ,round(np.std(coeffahb),4), 'and the error % is', abs(round(100*(np.std(coeffahb)/np.mean(coeffahb)),3)),'%')
#print('The mean value for vertical asymmetry a is', round(np.mean(coeffava),4), '+-' ,round(np.std(coeffava),4), 'and the error % is', abs(round(100*(np.std(coeffava)/np.mean(coeffava)),3)),'%')
#print('The mean value for vertical asymmetry b is', round(np.mean(coeffavb),4), '+-' ,round(np.std(coeffavb),4), 'and the error % is', abs(round(100*(np.std(coeffavb)/np.mean(coeffavb)),3)),'%')
#
#print('The mean value for horizontal moment a is', round(np.mean(coeffwha),4), '+-' ,round(np.std(coeffwha),4), 'and the error % is', abs(round(100*(np.std(coeffwha)/np.mean(coeffwha)),3)),'%')
#print('The mean value for horizontal moment b is', round(np.mean(coeffwhb),4), '+-' ,round(np.std(coeffwhb),4), 'and the error % is', abs(round(100*(np.std(coeffwhb)/np.mean(coeffwhb)),3)),'%')
#print('The mean value for vertical moment a is', round(np.mean(coeffwva),4), '+-' ,round(np.std(coeffwva),4), 'and the error % is', abs(round(100*(np.std(coeffwva)/np.mean(coeffwva)),3)),'%')
#print('The mean value for vertical moment b is', round(np.mean(coeffwvb),4), '+-' ,round(np.std(coeffwvb),4), 'and the error % is', abs(round(100*(np.std(coeffwvb)/np.mean(coeffwvb)),3)),'%')
binz=50

def supfunction(c,b):
    counts,bins, density = plt.hist(c, bins=b, density=True)
    bins_center = (bins[:-1]+bins[1:])/2
    mu, std = norm.fit(c)
    x = np.linspace(min(bins_center)-abs(min(bins_center)*0.2), max(bins_center)+abs(min(bins_center)*0.2), binz)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    fita = LinearRegression().fit(counts.reshape((-1, 1)), p)   
    r = fita.score(counts.reshape((-1, 1)), p)
    print('mu = ', round(mu,3), 'std =', round(std,3), 'and r2 =', round(r,3), '100*(std/mean) =', round(abs(100*(std/mu)),3))
    #np.savetxt('/home/andrea/PAPERS/JETREALTIME/Figure6/coeffaha.dat',coeffaha)
    #np.savetxt('/home/andrea/PAPERS/JETREALTIME/Figure6/coeffwhb_gaussian_x.dat',x)
    #np.savetxt('/home/andrea/PAPERS/JETREALTIME/Figure6/coeffwhb_gaussian_y.dat',p)

#plt.close('all')
#      #HISTOGRAM     
#           
plt.figure(400,figsize = (16, 9))
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1]) 

ax0 = plt.subplot(gs[0])
supfunction(coeffaha, binz)
plt.xlabel('$a_H$ asymmetry')
plt.ylabel('#')

ax1 = plt.subplot(gs[1])
supfunction(coeffava, binz)
plt.xlabel('$a_V$ asymmetry')
plt.ylabel('#')


ax2 = plt.subplot(gs[2])
supfunction(coeffwha, binz)
plt.xlabel('$a_H$ weighted average')
plt.ylabel('#')
#
#
ax3 = plt.subplot(gs[3])
supfunction(coeffwva, binz)
plt.xlabel('$a_V$ weighted average')
plt.ylabel('#')

ax4 = plt.subplot(gs[4])
supfunction(coeffahb, binz)
plt.xlabel('$b_H$ asymmetry')
plt.ylabel('#')

ax5 = plt.subplot(gs[5])
supfunction(coeffavb, binz)
plt.xlabel('$b_V$ asymmetry')
plt.ylabel('#')

ax6 = plt.subplot(gs[6])
supfunction(coeffwhb, binz)
plt.xlabel('$b_H$ weighted average')
plt.ylabel('#')
#
ax7 = plt.subplot(gs[7])
supfunction(coeffwvb, binz)
plt.xlabel('$b_V$ weighted average')
plt.ylabel('#')
           
           



#
#
#plt.figure(200,figsize = (16, 9))
#gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1]) 
#
#ax0 = plt.subplot(gs[0])
#plt.scatter(np.linspace(0,newlength, newlength),coeffaha, s=10, color='r')
#plt.ylabel('$a_H$ asymmetry')
#
#plt.setp(ax0.get_xticklabels(), visible=False)
#plt.subplots_adjust(hspace=.0)
#
#ax1 = plt.subplot(gs[1], sharex = ax0)
#plt.scatter(np.linspace(0,newlength, newlength),coeffava, s=10, facecolors='none', edgecolors='r')
#plt.ylabel('$a_V$ asymmetry')
#plt.subplots_adjust(hspace=.0)
#plt.setp(ax1.get_xticklabels(), visible=False)
#
#
#ax2 = plt.subplot(gs[2], sharex = ax0)
#plt.scatter(np.linspace(0,newlength, newlength),coeffwha, s=10, color='r')
#plt.ylabel('$a_H$ weighted average')
#plt.subplots_adjust(hspace=.0)
#plt.setp(ax2.get_xticklabels(), visible=False)
##
##
#ax3 = plt.subplot(gs[3], sharex = ax0)
#plt.scatter(np.linspace(0,newlength, newlength),coeffwva, s=10, facecolors='none', edgecolors='r')
#plt.ylabel('$a_V$ weighted average')
#plt.setp(ax3.get_xticklabels(), visible=False)
#plt.subplots_adjust(hspace=.0)
#
#ax4 = plt.subplot(gs[4])
#plt.scatter(np.linspace(0,newlength, newlength),coeffahb, s=10, color='b')
#plt.ylabel('$b_H$ asymmetry')
#plt.setp(ax0.get_xticklabels(), visible=False)
#plt.subplots_adjust(hspace=.0)
#
#
#ax5 = plt.subplot(gs[5], sharex = ax0)
#plt.scatter(np.linspace(0,newlength, newlength),coeffavb, s=10, facecolors='none', edgecolors='b')
#plt.ylabel('$b_V$ asymmetry')
#plt.subplots_adjust(hspace=.0)
#plt.setp(ax1.get_xticklabels(), visible=False)
#
#
#ax6 = plt.subplot(gs[6], sharex = ax0)
#plt.scatter(np.linspace(0,newlength, newlength),coeffwhb, s=10, color='b')
#plt.ylabel('$b_H$ weighted average')
#plt.subplots_adjust(hspace=.0)
#plt.setp(ax2.get_xticklabels(), visible=False)
#plt.xlabel('shot #')
##
#ax7 = plt.subplot(gs[7], sharex = ax0)
#plt.scatter(np.linspace(0,newlength, newlength),coeffwvb, s=10, facecolors='none', edgecolors='b')
#plt.ylabel('$b_V$ weighted average')
#plt.setp(ax3.get_xticklabels(), visible=False)
#plt.subplots_adjust(hspace=.0)
#plt.xlabel('shot #')
#  