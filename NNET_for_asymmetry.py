#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:11:35 2020

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
import random as rnd
from matplotlib import gridspec




def pure_linear(x):
    return x

def f(x):
    return np.int(x)
f2 = np.vectorize(f)

get_custom_objects().update({'pure_linear': pure_linear})



data_Train_Input = np.loadtxt('/home/andrea/JET/84786_84798_asimV.txt')     	#targets of the training set
data_Train_Input = data_Train_Input.reshape((-1, 1))
data_Train_Target =  np.loadtxt('/home/andrea/JET/84786_84798_efitV.txt') 		# inputs of the training set                      
data_Train_Target = data_Train_Target.reshape((-1, 1))


#% ------------------------------------------------------------------
#% Neural Network Layout
#% ------------------------------------------------------------------

data_Train_Input_norm = stats.zscore(data_Train_Input)
data_Train_Input_mean = np.mean(data_Train_Input)
data_Train_Input_std = np.std(data_Train_Input, ddof=1)


data_Train_Target_norm = stats.zscore(data_Train_Target)
data_Train_Target_mean = np.mean(data_Train_Target)
data_Train_Target_std = np.std(data_Train_Target, ddof=1)



#
#model_filename = "ModelZ_asymmetry.h5"
#
#if os.path.isfile(model_filename):
#
#    modelZ = load_model(model_filename)
#
#    print("Loaded model")
#
#else:
#
#    print('Create new model')
 
model_filename = "ModelR_asymmetry.h5"

if os.path.isfile(model_filename):

    model = load_model(model_filename)

    print("Loaded model")

else:

    print('Create new model')

modelZ = load_model("ModelZ_asymmetry.h5")

model = Sequential()
model.add(Input(shape=(1,)))
model.add(Dense(12,activation='relu', name="hidden_layer1"))
model.add(Dense(8,activation='relu', name="hidden_layer2"))
model.add(Dense(1, activation='pure_linear', name="final_layer"))

model.compile(loss='MeanSquaredError', optimizer= Adam(), metrics=['mae'])
model.summary()

callback1 = ModelCheckpoint(model_filename, save_best_only=True, verbose=True)
callback2 = EarlyStopping(monitor='loss', patience=500)

history = model.fit(data_Train_Input_norm, data_Train_Target_norm, 
          validation_split=0.5, epochs=2000, batch_size=256, callbacks=[callback1, callback2])


test_pulse_Z  = np.loadtxt('/home/andrea/JET/84536_asimH.txt')
target_pulse_Z  = np.loadtxt('/home/andrea/JET/84536_efitH.txt')
test_pulse_Z = test_pulse_Z.reshape((-1, 1))
test_pulse_Z_norm = stats.zscore(test_pulse_Z)
test_pulse_Z_mean = np.mean(test_pulse_Z)
test_pulse_Z_std = np.std(test_pulse_Z, ddof=1)

target_pulse_Z = target_pulse_Z.reshape((-1, 1))
target_pulse_Z_norm = stats.zscore(target_pulse_Z)
target_pulse_Z_mean = np.mean(target_pulse_Z)
target_pulse_Z_std = np.std(target_pulse_Z, ddof=1)

test_pulse_R = np.loadtxt('/home/andrea/JET/84536_asimV.txt')
target_pulse_R  = np.loadtxt('/home/andrea/JET/84536_efitV.txt')
test_pulse_R = test_pulse_R.reshape((-1, 1))
test_pulse_R_norm = stats.zscore(test_pulse_R)
test_pulse_R_mean = np.mean(test_pulse_R)
test_pulse_R_std = np.std(test_pulse_R, ddof=1)

target_pulse_R = target_pulse_R.reshape((-1, 1))
target_pulse_R_norm = stats.zscore(target_pulse_R)
target_pulse_R_mean = np.mean(target_pulse_R)
target_pulse_R_std = np.std(target_pulse_R, ddof=1)

prediction_test_Z_norm = modelZ.predict(test_pulse_Z_norm)
prediction_test_R_norm = model.predict(test_pulse_R_norm)


plt.close('all')





plt.figure(1)
plt.subplot(1,2,1)
# summarize history for accuracy
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



plt.figure(200,figsize = (17, 10))
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1]) 

ax0 = plt.subplot(gs[0])
plt.plot(prediction_test_Z_norm*target_pulse_Z_std + target_pulse_Z_mean,'r')
plt.plot((test_pulse_Z+0.5584572857142857)/2.2384285714285714 ,'b')
plt.plot(target_pulse_Z,'k')
plt.xlabel('time (au)')
plt.ylabel('Z (m)')
plt.legend(['NNET','linear fit', 'EFIT'])
plt.ylim(0.18, 0.36)  
plt.xlim(0, 410)
plt.show()
plt.setp(ax0.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.0)


ax1 = plt.subplot(gs[1], sharex = ax0)
plt.plot(100*target_pulse_Z - 100*(prediction_test_Z_norm*target_pulse_Z_std + target_pulse_Z_mean),'r--')
plt.plot(100*target_pulse_Z - 100*((test_pulse_Z+0.5584572857142857)/2.2384285714285714),'b--')
plt.plot(np.linspace(0, 410, 100), np.zeros((100)) ,'k--')
plt.xlabel('time (au)')
plt.ylabel('diff (cm)')
plt.legend(['NNET','linear fit'])
plt.ylim(-10, 10)
plt.xlim(0, 410)
plt.show()
plt.setp(ax1.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.0)



ax2 = plt.subplot(gs[2], sharex = ax0)
plt.plot(prediction_test_R_norm*target_pulse_R_std + target_pulse_R_mean,'r')
plt.plot((test_pulse_R+3.868863414285714)/1.2966 ,'b')
plt.plot(target_pulse_R,'k')
plt.xlabel('time (au)')
plt.ylabel('R (m)')
plt.ylim(2.85, 3.25) 
plt.xlim(0, 410)  
plt.show()
plt.setp(ax2.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.0)

ax3 = plt.subplot(gs[3], sharex = ax0)
plt.plot(100*target_pulse_R - 100*(prediction_test_R_norm*target_pulse_R_std + target_pulse_R_mean),'r--')
plt.plot(100*target_pulse_R - 100*((test_pulse_R+3.868863414285714)/1.2966),'b--')
plt.plot(np.linspace(0, 410, 100), np.zeros((100)) ,'k--')
plt.xlabel('time (au)')
plt.ylabel('diff (cm)')
plt.ylim(-10, 10)
plt.xlim(0, 410)
plt.show()
plt.subplots_adjust(hspace=.0)


