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

x_train,x_test,y_train,y_test=splitter(state,Energy,test_size=0.96) #Split data into train and test (96% test)

# %%

"""Define tunable model parameters"""

eta=np.logspace(-5,1,7)           #Define vector of learning rates (parameter to SGD optimiser)
lamda=np.logspace(-4,4,9)         #Define vector of hyperparameters 
n_layers=1                        #Define number of hidden layers in the model
n_neuron=(x_train.shape[1])**2    #Define number of neurons per layer in the model (L^2 here for simplicity)
epochs=100                        #Number of reiterations over the input data
batch_size=50                     #Number of samples per gradient update

# %%

"""Define custom metric"""

def R2_score(y_true,y_pred):
    SS_res=K.sum(K.square(y_true-y_pred)) 
    SS_tot=K.sum(K.square(y_true-K.mean(y_pred))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# %%

"""Define function to create Deep Neural Network Model using Keras"""

def NN_Model(n_layers,n_neuron,eta,lamda):
    model=Sequential()
    for i in range(n_layers):              #Run loop to add hidden layers to model
        model.add(Dense(n_neuron,activation='sigmoid',kernel_regularizer=regularizers.l1(lamda))) #Using l1
    model.add(Dense(1,activation='relu'))      #add output layer to model
    
    sgd=optimizers.SGD(lr=eta)    #Define optimiser for model (lr is learning rate)
    model.compile(loss='mean_squared_error',optimizer=sgd,metrics=[R2_score])
    return model
    
# %%
    
"""Perform regression on data)"""

R2_NN=np.zeros((len(lamda),len(eta)),dtype=object)    #Define 2D matrix to store R2 scores (variation with hyperparameters
                                                      #and learning rates)
R2_train=np.zeros((len(lamda),len(eta)))
R2_test=np.zeros((len(lamda),len(eta)))

for i in range(len(lamda)):
    for j in range(len(eta)):
        DNN_model=NN_Model(n_layers,n_neuron,eta[j],lamda[i])
        DNN_model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=0)
        R2_train[i,j]=DNN_model.evaluate(x_train,y_train)[1]
        R2_test[i,j]=DNN_model.evaluate(x_test,y_test)[1]
        
# %% 
        
"""Plot results (no Bootstrap framework)"""

plt.figure()
plt.semilogx(lamda,R2_train[:,0],'--g',label='Learning Rate=1E-5, Train Data')
plt.semilogx(lamda,R2_test[:,0],'-*g',label='Learning Rate=1E-5, Test Data')
plt.semilogx(lamda,R2_train[:,5],'--r',label='Learning Rate=1E+0, Train Data')
plt.semilogx(lamda,R2_test[:,5],'-*r',label='Learning Rate=1E+0, Test Data')
plt.legend()
plt.xlabel('Hyperparameters')
plt.ylabel('R2')
plt.title('Deep Neural Network Performance')
plt.show()







