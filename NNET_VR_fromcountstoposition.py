# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:04:10 2020

@author: andrea
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Input
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import get_custom_objects
from tensorflow.python.keras.models import load_model
from scipy import stats
from scipy import signal
from matplotlib import gridspec


#H_big = []
#V_big = []
#R_big = []
#Z_big = []
#

def pure_linear(x):
    return x

def f(x):
    return np.int(x)
f2 = np.vectorize(f)

def interpolation(newtime, oldtime, array):
    return np.interp(newtime, oldtime, array)

get_custom_objects().update({'pure_linear': pure_linear})
#
#
#shot=[84536,84539, 84540, 84541,84542,84543,84544,84545,84786,84788,84789,84792,84796,84797,84798,
#      85197, 85201, 85203, 85204, 85205,85393, 85394, 85399,85400, 85401, 85402, 85403,
#      92393, 92394, 92395, 92398, 92399,94256,94257,94259,94261,94262,94263,94264,94265,94266,94268,94270,
#      86658, 86659,86660,86661,86662,86663,86664,86667,86668,86669,86670,86671,86672,86673,86674,86675,86676,86677,86678,86679,
#      94665,94666,94667,94669,94670,94671,94672,94674,94675,94676,94677,94678,94679,94680,94681,94682,94683,94684,
#      85228	,85229	,85230	,85231	,85232	,86362	,86365	,86366	,86367	,86368	,86374	,86375	,86377	,
#      86378	,86379	,86380	,86381	,86382	,86383	,86386	,86388	,86389	,86390	,86392,	86399	,86401	,86404	,86407	,
#      86411	,86412	,86413	,86414	,86416	,86417	,86419	,86421	,86435	,86436	,86437	,86439	,86441	,86444	,86445,
#      86598	,86599	,86600	,86602	,86603	,86604	,86605	,86606,86982	,86983	,86985	,86986	,86987	,86988	,86989	,86991	,86992	,
#      86993	,86994, 87140	,87142	,87143	,87144,	87160	,87161	,87162	,87164,87228	,87231	,87232	,87233	,87240,87243	,87244	,87246	,
#      87247	,87248	,87249	,87250	,87251	,87252	,87253	,87267	,87268	,87271	,87273	,87276	,87277	,87278	,87280]
#
#shot = np.sort(shot)
#
#filt=351; countsleft=200 ; countsright=200


#for w in range(0, len(shot)):
#    plt.close('all')
#    print(shot[w])
#    
#    time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCtime_'+ str(shot[w])+'.txt')
#    f = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCdata_'+ str(shot[w])+'.txt')
#    refita = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/REFIT_'+ str(shot[w])+'.txt')
#    zefita = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ZEFIT_'+ str(shot[w])+'.txt')
#    time_maga = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/time_mag_'+ str(shot[w])+'.txt')
#    nbi_time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/nbi_time'+ str(shot[w])+'.txt')
#    nbi = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/nbi'+ str(shot[w])+'.txt')
#    #print('The integration time of the NC is', round(float(np.diff(time)[0]*1000)),'ms')
#    
#   
#    newtime = np.linspace(round(float((time[0]))), round(float((time[-1]))), (round(float((time[-1])))-round(float((time[0]))))*1000)
#
#        
#
#    counts_old = np.reshape(f,(len(time),20))   #NEW NEUTRON CAMERA
#    counts = np.zeros((len(newtime),20))
#    
#    if round(float(np.diff(time)[0]*1000)) > 1.5:
#        for i in range(len(counts_old[1])):
#            counts[:,i] = round(float(np.diff(time)[0]*1000))*(interpolation(newtime,time,counts_old[:,i])/1000)
#        time=newtime
#    else:
#        counts = np.zeros((len(time),20))
#        counts = counts_old
#        print('The time resolution is 1 ms, please go ahead with the analysis.')    
##    #counts = np.reshape(f,(len(time),19))/100  #OLD NEUTRON CAMERA
#
## Here I select one the entries with number of counts greater than a certain number, both for left ansd right time limit
#
#    timeindex = np.sum(counts,1)
#
#    idxa = (np.where((timeindex>countsleft)))
#    idxb = (np.where((timeindex>countsright)))
#    idxnbi = np.where((nbi>1e6))
#
#    a = time[idxa] 
#    b = time[idxb]
#
#    if a[0]>nbi_time[idxnbi][0]:
#        t1 = a[0] 
#    else:
#        t1 = nbi_time[idxnbi][0]
#        
#    if b[-1]<nbi_time[idxnbi][-1]:
#        t2 = b[-1] 
#    else:            
#        t2 = nbi_time[idxnbi][-1]
#   
#    t1=a[0]
#    t2=b[-1]
#   
#
##Here I define the length and the area of collimators
#
#    Len   = np.array((  1.615991444 ,  1.581272864 , 1.552087292 ,  1.532928955 , 1.522378565 ,  1.521778565 ,  1.533228955 , 1.553687292 ,  1.581572864 ,  1.617691444 ,  1.518333124 ,  1.515424142 ,  1.511876256 ,  1.510795114 ,  1.5093 ,  1.510695237 ,  1.514264589 ,  1.514830584 ,   1.518823781))
#    radii = np.array((        1.050 ,        1.050 ,       1.050 ,        1.050 ,       1.050 ,        1.050 ,        1.050 ,       1.050 ,        1.050 ,        1.050 ,        0.500 ,        0.600 ,        0.750 ,         0.85 ,   1.050 ,        1.050 ,        1.050 ,        1.050 ,         1.050))/100.0 
#    areas = np.pi * (radii**2) 
#
# #the horizontal camera has  10 channels (1-10)
## the vertical camera has 9 channels (11-19)
#    Ha = np.zeros((len(time),10))
#    Va = np.zeros((len(time),9))
#    Hchannel = np.linspace(1,10,10)
#    Vchannel = np.linspace(11,19,9)
#    Ha = counts[0:-1, 0:10]
#    Va = counts[0:-1, 10:19]
#    
#
# # HERE I CONVERTED FROM NEUTRON EMISSION RATE (n/s) TO NEUTRON EMISSIVITY (n/s/m^2)    
#    H = np.multiply(Ha,Len[0:10]**2) / (areas[0:10]**2)
#    V = np.multiply(Va,Len[10:19]**2) / (areas[10:19]**2)   
#    
#  
#    # HERE I SELECT ONLY THE ELEMENTS IN THE WANTED TIME WINDOW BECAUSE THERE ARE EMPTY SPACES
#
#    idx = (np.where((time>t1) & (time<t2)))
#    t = time[idx]
#    idx_efit = (np.where((time_maga>t1) & (time_maga<t2)))
#    time_mag = time_maga[idx_efit]
#    refit = refita[idx_efit]
#    zefit = zefita[idx_efit]
#    H = H[idx]
#    V = V[idx]
#    
#    H_big = np.append(H_big, H)
#    V_big = np.append(V_big, V)
#    R = np.interp(t,time_mag, refit)
#    Z = np.interp(t,time_mag, zefit)
#    R_big = np.append(R_big,R)
#    Z_big = np.append(Z_big,Z)
#    
#####
#H_big = np.reshape(H_big, (4813889,10))    
#V_big = np.reshape(V_big, (4813889,9))  
#np.savetxt('TRAINING_SET_NC_H_all.txt',H_big)
#np.savetxt('TRAINING_SET_NC_V_all.txt',V_big)
#np.savetxt('TRAINING_SET_EFIT_Z_all.txt',Z_big)
#np.savetxt('TRAINING_SET_EFIT_R_all.txt',R_big)

#data_Train_Input = np.loadtxt('TRAINING_SET_NC_H_all.txt')
#data_Train_Input_norm = stats.zscore(data_Train_Input)
#data_Train_Input_mean = np.mean(data_Train_Input)
#data_Train_Input_std = np.std(data_Train_Input, ddof=1)
#
#data_Train_Target = np.loadtxt('TRAINING_SET_EFIT_Z_all.txt')
#data_Train_Target_norm = stats.zscore(data_Train_Target)
#data_Train_Target_mean = np.mean(data_Train_Target)
#data_Train_Target_std = np.std(data_Train_Target, ddof=1)

#model_filename = "Model_from_counts_to_position_H_all.h5"

model_filename = 'Model_from_counts_to_position_V_small.h5'
if os.path.isfile(model_filename):

    model = load_model(model_filename)

    print("Loaded model")

else:

    print('Create new model')


#    
#model = Sequential()
#model.add(Input(shape=(10,)))
##model.add(Dense(32,activation='tanh', name="hidden_layer1"))
#model.add(Dense(16,activation='tanh', name="hidden_layer2"))
#model.add(Dense(8,activation='tanh', name="hidden_layer3"))
#model.add(Dense(1, activation='pure_linear', name="final_layer"))
##
#model.compile(loss='MeanSquaredError', optimizer= Adam(), metrics=['mae'])
#model.summary()
#
#callback1 = ModelCheckpoint(model_filename, save_best_only=True, verbose=True)
#callback2 = EarlyStopping(monitor='loss', mode='min', patience=100)
#
#history = model.fit(data_Train_Input_norm, data_Train_Target_norm, 
#                    validation_split=0.5,  epochs=1000, batch_size=len(data_Train_Target_norm), callbacks=[callback1, callback2])
#
#plt.figure(1)
#plt.subplot(1,2,1)
## summarize history for accuracy
#plt.plot(history.history['mae'])
#plt.plot(history.history['val_mae'])
#plt.title('model mae')
#plt.ylabel('mae')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
## summarize history for loss
#plt.subplot(1,2,2)
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
#

#### HERE I AM TESTING THE NNET #############


#
############# THESE ARE GREAT ##########
#shot = [84536,84539, 84540, 84541,84542,84543,84544,84545,84786,84788,84789,84792,84796,84797,84798,
#      85197, 85201, 85203, 85204, 85205,85393, 85394, 85399,85400, 85401, 85402, 85403,
#      92393, 92394, 92395, 92398, 92399,94256,94257,94259,94261,94262,94263,94264,94265,94266,94268,94270,
#      86658, 86659,86660,86661,86662,86663,86664,86667,86668,86669,86670,86671,86672,86673,86674,86675,86676,86677,86678,86679,
#      94665,94666,94667,94669,94670,94671,94672,94674,94675,94676,94677,94678,94679,94680,94681,94682,94683,94684,
#      85228	,85229	,85230	,85231	,85232	,86362	,86365	,86366	,86367	,86368	,86374	,86375	,86377	,
#      86378	,86379	,86380	,86381	,86382	,86383	,86386	,86388	,86389	,86390	,86392,	86399	,86401	,86404	,86407	,
#      86411	,86412	,86413	,86414	,86416	,86417	,86419	,86421	,86435	,86436	,86437	,86439	,86441	,86444	,86445,
#      86598	,86599	,86600	,86602	,86603	,86604	,86605	,86606,86982	,86983	,86985	,86986	,86987	,86988	,86989	,86991	,86992	,
#      86993	,86994, 87140	,87142	,87143	,87144,	87160	,87161	,87162	,87164,87228	,87231	,87232	,87233	,87240,87243	,87244	,87246	,
#      87247	,87248	,87249	,87250	,87251	,87252	,87253	,87267	,87268	,87271	,87273	,87276	,87277	,87278	,87280]
#shot = np.sort(shot)
#countsleft=200 ; countsright=200; 

shot = [84540];countsleft=200 ;countsright=200; filt=21



for w in range(0, len(shot)):
    plt.close('all')
    print(shot[w])
    
    time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCtime_'+ str(shot[w])+'.txt')
    f = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/NCdata_'+ str(shot[w])+'.txt')
    refita = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/REFIT_'+ str(shot[w])+'.txt')
    zefita = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/ZEFIT_'+ str(shot[w])+'.txt')
    time_maga = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/time_mag_'+ str(shot[w])+'.txt')
    nbi_time = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/nbi_time'+ str(shot[w])+'.txt')
    nbi = np.loadtxt('/media/andrea/My Passport Ultra/JET_REAL_TIME/Global/nbi'+ str(shot[w])+'.txt')
    print('The integration time of the NC is', round(float(np.diff(time)[0]*1000)),'ms')
#    if round(float(np.diff(time)[0]*1000))==10:
#        filt=11
#    elif round(float(np.diff(time)[0]*1000))==5:   
#        filt=61
#    elif round(float(np.diff(time)[0]*1000))==1:   
#        filt=111    
#    counts = np.reshape(f,(len(time),19))   #NEW NEUTRON CAMERA
    newtime = np.linspace(round(float((time[0]))), round(float((time[-1]))), (round(float((time[-1])))-round(float((time[0]))))*100)



    counts_old = np.reshape(f,(len(time),20))   #NEW NEUTRON CAMERA
    counts = np.zeros((len(time_maga),20))
    
    for i in range(len(counts_old[1])):
        counts[:,i] = round(float(np.diff(time_maga)[0]*1000))*interpolation(time_maga,time,counts_old[:,i])/round(float(np.diff(time)[0]*1000))
   

# Here I select one the entries with number of counts greater than a certain number, both for left ansd right time limit

    timeindex = np.sum(counts,1)

    idxa = (np.where((timeindex>countsleft)))
    idxb = (np.where((timeindex>countsright)))


    a = time_maga[idxa] 
    b = time_maga[idxb]
    idxnbi = np.where((nbi>1e6))
    if a[0]>nbi_time[idxnbi][0]:
        t1 = a[0] 
    else:
        t1 = nbi_time[idxnbi][0]
        
    if b[-1]<nbi_time[idxnbi][-1]:
        t2 = b[-1] 
    else:            
        t2 = nbi_time[idxnbi][-1]
   
    t1=44.09
    t2=48.04

#Here I define the length and the area of collimators

    Len   = np.array((  1.615991444 ,  1.581272864 , 1.552087292 ,  1.532928955 , 1.522378565 ,  1.521778565 ,  1.533228955 , 1.553687292 ,  1.581572864 ,  1.617691444 ,  1.518333124 ,  1.515424142 ,  1.511876256 ,  1.510795114 ,  1.5093 ,  1.510695237 ,  1.514264589 ,  1.514830584 ,   1.518823781))
    radii = np.array((        1.050 ,        1.050 ,       1.050 ,        1.050 ,       1.050 ,        1.050 ,        1.050 ,       1.050 ,        1.050 ,        1.050 ,        0.500 ,        0.600 ,        0.750 ,         0.85 ,   1.050 ,        1.050 ,        1.050 ,        1.050 ,         1.050))/100.0 
    areas = np.pi * (radii**2) 

 #the horizontal camera has  10 channels (1-10)
# the vertical camera has 9 channels (11-19)
    Ha = np.zeros((len(time),10))
    Va = np.zeros((len(time),9))
    Hchannel = np.linspace(1,10,10)
    Vchannel = np.linspace(11,19,9)
    Ha = counts[0:-1, 0:10]
    Va = counts[0:-1, 10:19]
    
 
 # HERE I CONVERTED FROM NEUTRON EMISSION RATE (n/s) TO NEUTRON EMISSIVITY (n/s/m^2)    
    H = np.multiply(Ha,Len[0:10]**2) / (areas[0:10]**2)
    V = np.multiply(Va,Len[10:19]**2) / (areas[10:19]**2)   
    
  
    # HERE I SELECT ONLY THE ELEMENTS IN THE WANTED TIME WINDOW BECAUSE THERE ARE EMPTY SPACES
    
    idx = (np.where((time_maga>t1) & (time_maga<t2)))
    t = time_maga[idx]
    time_mag = time_maga[idx]
    refit = refita[idx]
    zefit = zefita[idx]
    H = H[idx]
    V = V[idx]

    #V[:,7]=1e-26
    #V[:,8]=1e-26
    
    test_pulse  = V
    target_pulse  = refit

    test_pulse_norm = stats.zscore(test_pulse)
    test_pulse_mean = np.mean(test_pulse)
    test_pulse_std = np.std(test_pulse, ddof=1)
    
    target_pulse_norm = stats.zscore(target_pulse)
    target_pulse_mean = np.mean(target_pulse)
    target_pulse_std = np.std(target_pulse, ddof=1)
    
    prediction= model.predict(test_pulse_norm)
    predi = prediction*target_pulse_std+target_pulse_mean
    
    predia = predi#signal.savgol_filter(np.ravel(predi), filt, 4)

    plt.figure(200,figsize = (17, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
    
    ax0 = plt.subplot(gs[0])
    plt.plot(t,predia,'r')
    plt.plot(t,target_pulse,'k')
    plt.xlabel('time (s)')
    plt.ylabel('R (m)')
    plt.legend(['NNET','EFIT'])
    plt.ylim(2.75, 3.2)  
    plt.show()
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=.0)
    
    
    ax1 = plt.subplot(gs[1], sharex = ax0)
    plt.plot(t,100*(target_pulse.reshape(-1,1) - predia.reshape(-1,1)),'r--')
    plt.plot(t, np.zeros((len(t))) ,'k--')
    plt.xlabel('time (s)')
    plt.ylabel('diff (cm)')
    plt.show()
    plt.subplots_adjust(hspace=.0)
    plt.ylim(-5, 5)
    plt.plot(t, 1.5*np.ones((len(t))), 'k')
    plt.plot(t, -1.5*np.ones((len(t))), 'k')

    plt.savefig('www' + str(shot[w]) + '_NNET_.png')

    plt.close('all')



#deltaR = predia.reshape(-1,1) -target_pulse.reshape(-1,1)

#stdR = np.sum((deltaR-np.mean(deltaR))**2)/len(deltaR)

#print('Delta R is ',round(np.mean(deltaR)*100,5) ,' +- ',round(stdR*100,5),' cm')    
    