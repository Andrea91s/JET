import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def showlines(xaxis, lines, xl='Bins', yl='ADC codes', title=' '):
# lines of form: s[1:],c[n,1:]
 #    print np.shape(xaxis), np.shape(lines)
     plt.figure()
     plt.xlabel(xl)
     plt.ylabel(yl)
     plt.title(title)
     nlines = np.shape(lines)[0]
     for n in range(nlines):
        plt.plot(xaxis,lines[n,:])
     plt.show()

      #  S00, alpha, edge, R0, Z0, kn1
  #  y = LOS(  x[3], x[4], E0, Ra, Za, Ea, minor, x[0], x[1], x[2]) * att * areas * areas * eff / 4.0 / np.pi / Len /Len - exper + bs

def LOS( R0, Z0, E0, Ra, Za, Ea, minor, S0, alpha, edge):
    if verbose: print (R0, Z0, E0, Ra, Za, Ea, minor, S0, alpha, edge)
    nrho  = 400                 
    delta = 1.0/nrho
    d2    = delta/2.0
    rh    = np.linspace(0, 1, nrho+1)
    rhin  = np.linspace(d2, 1.0-d2, nrho)
#    Ch    = np.array((           1 ,            2 ,           3 ,            4 ,           5 ,            6 ,            7 ,           8 ,            9 ,           10 ,           11 ,           12 ,           13 ,           14 ,      15 ,           16 ,           17 ,           18 ,           19 ))
    Rc    = np.array((        6.055 ,        6.055 ,       6.055 ,        6.055 ,       6.055 ,        6.055 ,        6.055 ,       6.055 ,        6.055 ,        6.055 ,       3.0215 ,       3.0215 ,       3.0215 ,       3.0215 ,  3.0215 ,       3.0215 ,       3.0215 ,       3.0215 ,        3.0215))
    Zc    = np.array((           0. ,           0. ,          0. ,           0. ,          0. ,           0. ,           0. ,          0. ,           0. ,           0. ,        3.442 ,        3.442 ,        3.442 ,        3.442 ,   3.442 ,        3.442 ,        3.442 ,        3.442 ,         3.442))
    uZ    = np.array((  0.319273584 ,  0.253538472 ,  0.18391704 ,  0.111798472 , 0.037376245 , -0.037376245 , -0.111798472 , -0.18391704 , -0.253538472 , -0.319273584 , -0.976664782 , -0.986666067 , -0.993983916 , -0.998488138 ,      -1 , -0.998488138 , -0.993983916 , -0.986666067 ,  -0.976664782))
    uR    = np.array(( -0.947662587 , -0.967325304 , -0.98294177 ,   -0.9937309 ,-0.999301264 , -0.999301264 ,   -0.9937309 , -0.98294177 , -0.967325304 , -0.947662587 , -0.214769419 ,  -0.16275771 , -0.109526134 , -0.054967607 ,       0 ,  0.054967607 ,  0.109526134 ,   0.16275771 ,   0.214769419))

    vols = np.zeros(nrho)
    Rx   = R0  + rh * (Ra-R0)
    Zx   = Z0  + rh * (Za-Z0)
    Ex   = E0  + rh * (Ea-E0)
#  print Rx
    vols = 2*np.pi*Rx*np.pi*Ex*minor*minor*rh*rh
    segs = np.diff(vols)

 #   print vols, segs, s2
    lens = np.zeros((19,nrho))
    for i in range(19):
   #     print
        delR = Rc[i] - Rx
        delZ = Zc[i] - Zx   
        a    = uR[i]*uR[i] + uZ[i]*uZ[i]/Ex/Ex
        b    = 2.0*(uR[i]*delR+uZ[i]*delZ/Ex/Ex)
        c    = delR*delR + delZ*delZ/Ex/Ex -minor*minor*rh*rh
        rt   = b*b-4*a*c
        ind  = np.where(rt>0)
        lamb = 2.0*np.sqrt(rt[ind])/a[ind]
        L    = nrho - len(ind[0]) + 1
        lens[i,L:] = np.diff(lamb)

    #    print i,L,lamb,np.diff(lamb),lamb[L-1]
    #    print i, lens[i,33:] 
 #   print S0,edge,rhin,alpha
 #   print (1-rhin*rhin)**alpha
    emiss = S0*((1-edge)*((1-rhin*rhin)**alpha) + edge)*1e15
    em    = emiss*segs
    stot  = em.sum()

    if verbose: print ('Total yield =', stot) 
    LI    = np.dot( lens, emiss)
 #   print LI
    return LI #, stot

def hurl(x):
      #  S00, alpha, edge, R0, Z0, edge
    y = LOS(x[3], x[4], E0, Ra, Za, Ea, minor, x[0], x[1], x[2]) * att * areas * areas * eff / 4.0 / np.pi / Len /Len - exper + bs
 #   print x,y,bs
    return y

# KN3 Geometry
Len   = np.array((  1.615991444 ,  1.581272864 , 1.552087292 ,  1.532928955 , 1.522378565 ,  1.521778565 ,  1.533228955 , 1.553687292 ,  1.581572864 ,  1.617691444 ,  1.518333124 ,  1.515424142 ,  1.511876256 ,  1.510795114 ,  1.5093 ,  1.510695237 ,  1.514264589 ,  1.514830584 ,   1.518823781))
radii = np.array((        1.050 ,        1.050 ,       1.050 ,        1.050 ,       1.050 ,        1.050 ,        1.050 ,       1.050 ,        1.050 ,        1.050 ,        0.500 ,        0.600 ,        0.750 ,         0.85 ,   1.050 ,        1.050 ,        1.050 ,        1.050 ,         1.050))/100.0 
areas = np.pi * radii * radii 
exper = np.array((4096,10446,17951,24735,24315,12904,5682,2280,1033,526,90,580,3990,13372,31759,33511,21845,7898,2920))
bscat = np.array((1e-13,1e-13,1e-13,1e-13,1e-13,1e-13,1e-13,1e-13,1e-13,1e-13,1e-13,1e-13,1e-13,1e-13,1e-13,1e-13,1e-13,1e-13,1e-13))

# Flux surfaces

R0    = 3.
Z0    = 0.3
E0    = 1.1
Ra    = 2.9
Za    = 0.1
Ea    = 1.7
minor = 1

# Camera related stuff

dt    = 0.5
eff   = 0.0157
att   = 0.92
kn1   = 9.92E15 * dt


# Emissivity profile

alpha = 5.005
edge  = 0.05
S00   = 0.3

bs =  10000*kn1*bscat*areas/3
verbose = False
 
x = np.array((S00, alpha, edge, R0, Z0 ))
fitted = optimize.least_squares(hurl, x)
print ('Fitted coeff:',fitted['x'])
verbose = True
ccc = hurl(fitted['x'])
LI  = ccc + exper + bs
print ('KN1: ', kn1)
# print 'Fitted yield:', yld 
lines = np.zeros((2,19))
lines[0,:] = exper
lines[1,:] = LI
xv = np.linspace(1,19,19)
showlines(xv, lines)    