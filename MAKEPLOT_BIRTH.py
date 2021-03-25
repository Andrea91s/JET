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
#plt.close('all')

def niceplot(x,y,a):
    data = np.zeros((len(x),2)) 
    data[:,0]=x
    data[:,1]=y
    k = kde.gaussian_kde(data.T)
    nbins=100
    xi,yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]  
    zi= k(np.vstack([xi.flatten(), yi.flatten()]))
    
    print(xi.min())
    print(xi.max())
    print(yi.min())
    print(yi.max())
    plt.figure(str(a))
    #plt.pcolormesh(xi,yi,zi.reshape(xi.shape),cmap=cm.hot_r)
    plt.scatter(x,y)
    fig=zi.reshape(xi.shape)
    #plt.xlim(1.9, 3.9)
    #plt.ylim(-1.5, 2)
#    if   a==1:
#        np.savetxt('/home/andrea/PAPERS/JETREALTIME/Figure11/Z.dat',fig) 
#    elif a==2:
#        np.savetxt('/home/andrea/PAPERS/JETREALTIME/Figure11/Ra.dat',fig) 
#    elif a==3:
    np.savetxt('/home/andrea/PAPERS/JETREALTIME/Figure11/0w.dat',fig)
#    elif a==4:
#        np.savetxt('/home/andrea/PAPERS/JETREALTIME/Figure2/Rm.dat',fig)     
    #plt.contour(xi,yi,zi.reshape(xi.shape),'k')

z=np.loadtxt('/home/andrea/TRANSP/94665_birth0_z.txt')

r=np.loadtxt('/home/andrea/TRANSP/94665_birth0_r.txt')



niceplot(r,z,a=2)
