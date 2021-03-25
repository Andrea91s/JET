# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:04:10 2020

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import gridspec
import random as rnd
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=5.0, facecolor='none', **kwargs):

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    print(cov)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    print(pearson)
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

refit1 = np.loadtxt('S1_REFIT.txt')
zefit1 = np.loadtxt('S1_ZEFIT.txt')
rasym1 = np.loadtxt('S1_VASIM.txt')
zasym1 = np.loadtxt('S1_HASIM.txt')

refit2 = np.loadtxt('S2_REFIT.txt')
zefit2 = np.loadtxt('S2_ZEFIT.txt')
rasym2 = np.loadtxt('S2_VASIM.txt')
zasym2 = np.loadtxt('S2_HASIM.txt')

refit3 = np.loadtxt('S3_REFIT.txt')
zefit3 = np.loadtxt('S3_ZEFIT.txt')
rasym3 = np.loadtxt('S3_VASIM.txt')
zasym3 = np.loadtxt('S3_HASIM.txt')

refit4 = np.loadtxt('S4_REFIT.txt')
zefit4 = np.loadtxt('S4_ZEFIT.txt')
rasym4 = np.loadtxt('S4_VASIM.txt')
zasym4 = np.loadtxt('S4_HASIM.txt')

refit5 = np.loadtxt('S5_REFIT.txt')
zefit5 = np.loadtxt('S5_ZEFIT.txt')
rasym5 = np.loadtxt('S5_VASIM.txt')
zasym5 = np.loadtxt('S5_HASIM.txt')

refit6 = np.loadtxt('S6_REFIT.txt')
zefit6 = np.loadtxt('S6_ZEFIT.txt')
rasym6 = np.loadtxt('S6_VASIM.txt')
zasym6 = np.loadtxt('S6_HASIM.txt')

refit7 = np.loadtxt('S7_REFIT.txt')
zefit7 = np.loadtxt('S7_ZEFIT.txt')
rasym7 = np.loadtxt('S7_VASIM.txt')
zasym7 = np.loadtxt('S7_HASIM.txt')

refit = np.hstack((refit1,refit2,refit3,refit4, refit5, refit6, refit7)).ravel()
zefit = np.hstack((zefit1,zefit2,zefit3, zefit4, zefit5, zefit6, zefit7)).ravel()
rasym= np.hstack((rasym1,rasym2,rasym3,rasym4, rasym5, rasym6, rasym7)).ravel()
zasym= np.hstack((zasym1,zasym2,zasym3,zasym4, zasym5, zasym6, zasym7)).ravel()



plt.close('all')
fitaH = LinearRegression().fit(zefit.reshape((-1, 1)), zasym)
fitaV = LinearRegression().fit(refit.reshape((-1, 1)), rasym)

r_sqaH = fitaH.score(zefit.reshape((-1, 1)), zasym)
r_sqaV = fitaV.score(refit.reshape((-1, 1)), rasym)

Z = np.linspace(np.min(zefit), np.max(zefit), 1000)
R = np.linspace(np.min(refit), np.max(refit), 1000)


fig, axs = plt.subplots(1,1, figsize=(9, 3))
axs.text(np.max(zefit), np.min(zasym), '$R^2$ = (%.3f)'%r_sqaH,
         verticalalignment='bottom', horizontalalignment='right', fontsize=25) 
axs.plot(Z, fitaH.intercept_+Z*fitaH.coef_, c='r', linewidth=3)
axs.scatter(zefit, zasym, c='b', s=10)
confidence_ellipse(zefit, zasym,  axs,n_std=3, edgecolor='red')

fig, axs2 = plt.subplots(1,1, figsize=(9, 3))
axs2.text(np.max(refit), np.min(rasym), '$R^2$ = (%.3f)'%r_sqaV,
         verticalalignment='bottom', horizontalalignment='right', fontsize=25) 
axs2.plot(R, fitaV.intercept_+R*fitaV.coef_, c='r', linewidth=3)
axs2.scatter(refit, rasym, c='b', s=10)
confidence_ellipse(refit, rasym,  axs2,n_std=3, edgecolor='red')
#plt.xlabel('Z (m)', fontsize=20)
#plt.ylabel('Asymmetry H', fontsize=20)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#
#plt.subplot(1,2,2)
#plt.text(np.max(refit), np.min(rasym), '$R^2$ = (%.3f)'%r_sqaV,
#         verticalalignment='bottom', horizontalalignment='right', fontsize=25) 
#plt.plot(R, fitaV.intercept_+ R*fitaV.coef_, c='b', linewidth=3)
#plt.scatter(refit, rasym, c='r', s=10)
#plt.xlabel('R (m)', fontsize=20)
#plt.ylabel('Asymmetry V', fontsize=20)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)

#plt.figure(500,figsize = (17, 8))
#plt.subplot(1,2,1)
#plt.scatter(zefit1, zasym1, c='r', s=10)
#plt.scatter(zefit2, zasym2, c='b', s=10)
#plt.scatter(zefit3, zasym3, c='y', s=10)
#plt.scatter(zefit4, zasym4, c='m', s=10)
#plt.scatter(zefit5, zasym5, c='g', s=10)
#plt.scatter(zefit6, zasym6, c='k', s=10)
#plt.scatter(zefit7, zasym7, c='c', s=10)
#plt.legend(['S1','S2','S3','S4','S5','S6','S7'],fontsize=16,frameon=False)
#
#plt.xlabel('Z (m)', fontsize=20)
#plt.ylabel('Asymmetry H', fontsize=20)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#
#plt.subplot(1,2,2)
#plt.scatter(refit1, rasym1, c='r', s=10)
#plt.scatter(refit2, rasym2, c='b', s=10)
#plt.scatter(refit3, rasym3, c='y', s=10)
#plt.scatter(refit4, rasym4, c='m', s=10)
#plt.scatter(refit5, rasym5, c='g', s=10)
#plt.scatter(refit6, rasym6, c='k', s=10)
#plt.scatter(refit7, rasym7, c='c', s=10)
#plt.xlabel('R (m)', fontsize=20)
#plt.ylabel('Asymmetry V', fontsize=20)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)

