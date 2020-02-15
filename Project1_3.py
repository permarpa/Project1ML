# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:41:47 2020

@author: Bharat Mishra, Patricia Perez-Martin, Pinelopi Christodoulou
"""

"""Project 1 - Ising Model - Part (d)"""

# %%

"""Import all necessary Python libraries (Neural Network Implementation using Keras alone)"""

from keras.models import Sequential     #This allows appending layers to existing models
from keras.layers import Dense          #This allows defining the characteristics of a particular layer
from keras import regularizers          #This allows using whichever regularizer we want (l1,l2,l1_l2)
from keras import optimizers            #This allows using whichever optimiser we want (sgd,adam,RMSprop)
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split as splitter
from sklearn.metrics import r2_score

# %%

"""Generate 1D Ising Dataset and split into training and testing data"""

np.random.seed(12)      

#Define system parameters

L=40                        #total 40 spins
n=int(1e+4)                 #total 10000 random spin configurations

state=np.random.choice([-1,1],size=(n,L))    #Create 10000 random spin states

#Define function to calculate system energy

def Ising_Energy(state):
    L=state.shape[1]            #Extract number of columns (spins)
    n=state.shape[0]            #Extract number of rows (spin configurations)
    J=-1.0               
    E=np.zeros(n)               
    for i in range(n):
        for j in range(L-1):
            if (j==0):
               E[i]+=J*state[i,j]*state[i,L-1]
            E[i]+=J*state[i,j]*state[i,j+1]
    return E

#Define system energy vector (training+testing data)
    
Energy=Ising_Energy(state)          #y values

x_train,x_test,y_train,y_test=splitter(state,Energy,test_size=0.20) #Split data into train and test (96% test)
y_train=np.reshape(y_train,(len(y_train),1))
y_test=np.reshape(y_test,(len(y_test),1))

# %%

"""Define tunable model parameters"""

eta=np.logspace(-3,-1,3)           #Define vector of learning rates (parameter to SGD optimiser)
lamda=np.logspace(-4,-2,3)         #Define vector of hyperparameters 
n_layers=2                         #Define number of hidden layers in the model
#n_neuron=(x_train.shape[1])       #Define number of neurons per layer in the model (L here for simplicity)
n_neuron=128
epochs=50                         #Number of reiterations over the input data
batch_size=50                     #Number of samples per gradient update

# %%

"""Define custom metric"""

def R2_score(y_true,y_pred):
    SS_res=K.sum(K.square(y_true-y_pred)) 
    SS_tot=K.sum(K.square(y_true-K.mean(y_pred))) 
    return ( 1 - SS_res/(SS_tot) )

# %%

"""Define function to create Deep Neural Network Model using Keras"""

def NN_Model(n_layers,n_neuron,eta,lamda):
    model=Sequential()
    for i in range(n_layers):              #Run loop to add hidden layers to model
        if (i==0):                         #First layer requires input dimensions = 40 spins
            model.add(Dense(n_neuron,activation='relu',kernel_regularizer=regularizers.l2(lamda),input_dim=40))
        else:                              #Corresponding layers are capable of automatic shape inference
            model.add(Dense(n_neuron,activation='relu',kernel_regularizer=regularizers.l2(lamda))) #Using l2
    model.add(Dense(1,activation='relu'))      #add output layer to model
    sgd=optimizers.SGD(lr=eta)    #Define optimiser for model (lr is learning rate)
    model.compile(loss='mean_squared_error',optimizer=sgd)   #can add metric here, but not necessary
    return model
    
# %%
    
"""Perform regression on data"""

R2_train=np.zeros((len(lamda),len(eta)))     #Define vector to store R2 metric for training data
R2_test=np.zeros((len(lamda),len(eta)))      #Define vector to store R2 metric for testing data

for i in range(len(lamda)):                  #Run loops over hyperparamaters and learning rates
    for j in range(len(eta)):
        DNN_model=NN_Model(n_layers,n_neuron,eta[j],lamda[i])   #Call model for each lamda and eta
        DNN_model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1)  #training on data
        y_train_pred=DNN_model.predict(x_train)       #predict y_train values
        y_test_pred=DNN_model.predict(x_test)         #predict y_test values
        R2_train[i,j]=r2_score(y_train,y_train_pred)  #calculate R2 scores 
        R2_test[i,j]=r2_score(y_test,y_test_pred)
        
# %% 
        
"""Plot results (no Bootstrap framework)"""

plt.figure()
plt.semilogx(lamda,R2_train[:,0],'--g',label='Learning Rate=1E-3, Train Data')
plt.semilogx(lamda,R2_test[:,0],'-*g',label='Learning Rate=1E-3, Test Data')
plt.semilogx(lamda,R2_train[:,1],'--r',label='Learning Rate=1E-2, Train Data')
plt.semilogx(lamda,R2_test[:,1],'-*r',label='Learning Rate=1E-2, Test Data')
plt.semilogx(lamda,R2_train[:,2],'--b',label='Learning Rate=1E-1, Train Data')
plt.semilogx(lamda,R2_test[:,2],'-*b',label='Learning Rate=1E-1, Test Data')
plt.legend()
plt.xlabel('Hyperparameters')
plt.ylabel('R2')
plt.title('Deep Neural Network Performance')
plt.show()







