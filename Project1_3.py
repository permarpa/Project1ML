# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:41:58 2020

@author: Bharat Mishra, Patricia Perez-Martin, Pinelopi Christodoulou
"""

"""Project 1 - Ising Model - Part (d)"""

# %%

"""Import all necessary Python libraries (Neural Network Implementation using Keras alone)"""

from keras.models import Sequential     #This allows appending layers to existing models
from keras.layers import Dense          #This allows defining the characteristics of a particular layer
from keras import regularizers          #This allows using whichever regularizer we want (l1,l2,l1_l2)
from keras import optimizers            #This allows using whichever optimiser we want (sgd,adam,RMSprop)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as splitter
from sklearn.metrics import r2_score

# %%

"""Generate 1D Ising Dataset and split into training and testing data"""

np.random.seed(12)      

#Define system parameters

L=40                        #total 40 spins
n=int(1e+4)                 #total 10000 random spin configurations

state=np.random.choice([-1,1],size=(n,L))    #Create 2D matrix of 10000 rows (representing 10000 spin
                                             #configurations) and 40 columns (representing 40 spins) 

#Define function to calculate system energy

def Ising_Energy(state):
    L=state.shape[1]            #Extract number of columns (spins)
    n=state.shape[0]            #Extract number of rows (spin configurations)
    J=-1.0               
    E=np.zeros(n)               
    for i in range(n):
        for j in range(L-1):
            if (j==0):          #Takes into consideration circularity
               E[i]+=J*state[i,j]*state[i,L-1]
            E[i]+=J*state[i,j]*state[i,j+1]
    return E

#Define system energy vector (training+testing data)
    
Energy=Ising_Energy(state)          #1D array of 10000 columns, one for each spin configuration

# %%

"""Reconstruct the input in the form of pair-wise interactions"""

def Design_Matrix(state):              #Design matrix consists of all 2 spin interactions
    L=state.shape[1]                   #Number of columns = spins
    n=state.shape[0]                   #Number of rows = spin configurations
    X=np.zeros((n,L**2))               #Define the design matrix
    for i in range(X.shape[0]):        #Run loop to add all 2-body interactions
        j=0
        for k in range(L):
            for l in range(L):
                X[i,j]=int(state[i,k]*state[i,l])
                j+=1
    return X                    #Return design matrix

# %%
    
"""Split the data into training and testing sets"""

X=Design_Matrix(state)

x_train,x_test,y_train,y_test=splitter(X,Energy,test_size=0.96) #Split data into train and test
y_train=np.reshape(y_train,(len(y_train),1))
y_test=np.reshape(y_test,(len(y_test),1))

# %%

"""Define tunable model parameters"""

eta=np.logspace(-3,-2,2)           #Define vector of learning rates (parameter to SGD optimiser)
lamda=np.logspace(-8,-1,8)         #Define vector of hyperparameters 
n_neuron=40                        #Number of neurons (not important here)
epochs=50                          #Number of reiterations over the input data (for minimising loss)
batch_size=100                     #Number of samples per gradient update


# %%

"""Define function to create Deep Neural Network Model using Keras"""

def NN_Model(inputsize,n_neuron,eta,lamda):
    model=Sequential()
    model.add(Dense(1,activation='linear',kernel_regularizer=regularizers.l2(lamda),input_dim=inputsize)) 
    #One input and one output layer, no hidden layer (weights matrix directly gives learned interaction)
    sgd=optimizers.SGD(lr=eta)    #Define optimiser for model (lr is learning rate)
    model.compile(loss='mean_squared_error',optimizer=sgd)   #can add metric here, but not necessary
    return model
    
# %%
    
"""Perform regression on data"""

R2_train=np.zeros((len(lamda),len(eta)))     #Define vector to store R2 metric for training data
R2_test=np.zeros((len(lamda),len(eta)))      #Define vector to store R2 metric for testing data

for i in range(len(lamda)):                  #Run loops over hyperparamaters and learning rates
    for j in range(len(eta)):
        DNN_model=NN_Model(x_train.shape[1],n_neuron,eta[j],lamda[i])   #Call model for each lamda and eta
        DNN_model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1)  #training on data
        if (i==1 and j==1):
            for layers in DNN_model.layers:
                weights=layers.get_weights()
        y_train_pred=DNN_model.predict(x_train)       #predict y_train values
        y_test_pred=DNN_model.predict(x_test)         #predict y_test values
        R2_train[i,j]=r2_score(y_train,y_train_pred)  #calculate R2 scores for training data
        R2_test[i,j]=r2_score(y_test,y_test_pred)     #calculate R2 scores for testing data
 
# %% 
        
"""Plot results (no Bootstrap framework)"""

plt.figure()
plt.semilogx(lamda,R2_train[:,0],'--g',linewidth=1.5,label='Learning Rate=1E-3, Train Data')
plt.semilogx(lamda,R2_test[:,0],'-*g',linewidth=1.5,label='Learning Rate=1E-3, Test Data')
plt.semilogx(lamda,R2_train[:,1],'--r',linewidth=1.5,label='Learning Rate=1E-2, Train Data')
plt.semilogx(lamda,R2_test[:,1],'-*r',linewidth=1.5,label='Learning Rate=1E-2, Test Data')
plt.legend()
plt.grid()
plt.xlabel('\u03BB',fontweight='bold')
plt.ylabel('$r^2$')
plt.title('DNN Performance (No Hidden Layer, Activation = linear)',fontweight='bold')
plt.show()

ticks=np.linspace(0,40,9,dtype=int)

plt.figure()
IntMat=np.reshape(weights[0],(L,L))
ax=sns.heatmap(IntMat,vmin=-1.0,vmax=1.0,cmap='seismic')
plt.xticks(ticks,ticks)
plt.yticks(ticks,ticks)
plt.title('Coupling Constants, \u03BB=1e-7,\u03B7=0.01, Normalised',fontweight='bold')
plt.show()

plt.figure()
IntMat=np.reshape(weights[0],(L,L))
ax=sns.heatmap(IntMat,cmap='seismic')
plt.xticks(ticks,ticks)
plt.yticks(ticks,ticks)
plt.title('Coupling Constants, \u03BB=1e-7,\u03B7=0.01, True',fontweight='bold')
plt.show()
