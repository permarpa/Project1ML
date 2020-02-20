# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 12:45:52 2020

@author: Bharat Mishra, Patricia Perez-Martin, Pinelopi Christodoulou
"""

"""Project 1 - Ising Model - part (e)"""

# %%

"""Import all necessary Python libraries (Neural Network implementation using Keras alone)"""

from keras.models import Sequential      #This allows appending layers to existing models
from keras.layers import Dense           #This allows defining the characteristics of a particular layer
from keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as splitter
import pickle
import os

# %%

"""Read data from IsingFiles"""

#Define path names for 2D Ising configuration and class labels

L=40                              #Define number of spins in the system
T=np.linspace(0.25,4.00,16)       #Define system temperatures
T=np.reshape(T,(len(T),1))
n=int(1e+4)                          #Define number of system configurations for each temperature 

datafile='E:\\My_Books\\Study\\UniCaen_Sem3\\Machine_Learning\\Project1\\IsingData\\'
labelfile='E:\\My_Books\\Study\\UniCaen_Sem3\\Machine_Learning\\Project1\\IsingData\\Ising2DFM_reSample_L40_T=All_labels.pkl'

for i in range(len(T)):
    pickle_off1=open(datafile+'Ising2DFM_reSample_L40_T=%.2f.pkl'%T[i],"rb")
    pickle_off2=open(labelfile,"rb")
    temp1=pickle.load(pickle_off1)                #Read data from datafile path name (2D spin configurations)
    temp1=np.unpackbits(temp1).reshape(-1,1600)   #Decompress and reshape array
    temp1=temp1.astype(int)                       #Set data as integers alone (spins can be +1 or -1)
    temp1[np.where(temp1==0)]=-1                  #Set all 0 spins to -1
    if (i==0):
        config=temp1
    else:
        config=np.concatenate((config,temp1),axis=0)

temp2=pickle.load(pickle_off2)                    #Read data from labelfile path name (phase labels for each configuration)  
temp2=np.reshape(temp2,(n*len(T),1))              #Reshape array 
temp2=temp2.astype(int)                           #Set data as integers alone (labels can be 0 or 1)
label=temp2
pickle_off1.close()
pickle_off2.close()

del temp1,temp2

# %%

"""Display data"""

plt.figure()                          #Plot one of the ordered lattices (any row from 0 to 69999)
temp=config[9999,:]
order_config_ex=np.reshape(temp,(L,L))
ax=sns.heatmap(order_config_ex,vmin=-1.0,vmax=1.0,cmap='plasma_r')
plt.xticks(np.linspace(0,40,9,dtype=int),np.linspace(0,40,9,dtype=int))
plt.yticks(np.linspace(0,40,9,dtype=int),np.linspace(0,40,9,dtype=int))
plt.title('Ordered Lattice',fontweight='bold')
plt.show()

plt.figure()                          #Plot one of the critical lattices (any row from 70000 to 99999)
temp=config[70000,:]
critical_config_ex=np.reshape(temp,(L,L))
ax=sns.heatmap(critical_config_ex,vmin=-1.0,vmax=1.0,cmap='plasma_r')
plt.xticks(np.linspace(0,40,9,dtype=int),np.linspace(0,40,9,dtype=int))
plt.yticks(np.linspace(0,40,9,dtype=int),np.linspace(0,40,9,dtype=int))
plt.title('Critical Lattice',fontweight='bold')
plt.show()

plt.figure()                          #Plot one of the disordered lattices (any row from 100000 to 159999)
temp=config[159999,:]
disorder_config_ex=np.reshape(temp,(L,L))
ax=sns.heatmap(disorder_config_ex,vmin=-1.0,vmax=1.0,cmap='plasma_r')
plt.xticks(np.linspace(0,40,9,dtype=int),np.linspace(0,40,9,dtype=int))
plt.yticks(np.linspace(0,40,9,dtype=int),np.linspace(0,40,9,dtype=int))
plt.title('Disordered Lattice',fontweight='bold')
plt.show()

del temp

# %%

"""Generate training and testing datasets (no criticality in testing)"""
#No criticality implies all configurations from 2.00 to 2.50 are taken as only training data and not testing

X_ordered=config[:70000,:]         #Collect all ordered spin configurations
Y_ordered=label[:70000,0]          #Collect all ordered labels

X_critical=config[70000:100000,:]    #Collect all critical spin configurations
Y_critical=label[70000:100000,0]     #Collect all critical labels

X_disordered=config[100000:,:]      #Collect all disorderd spin configurations
Y_disordered=label[100000:,0]       #Collect all disordered labels

X_temp=np.concatenate((X_ordered,X_disordered),axis=0)    #Combine ordered and disordered spins for split
Y_temp=np.concatenate((Y_ordered,Y_disordered),axis=0)    #Combine ordered and disordered labels for split

X_train,X_test,y_train,y_test=splitter(X_temp,Y_temp,test_size=0.5)

del X_temp,Y_temp

X_train=np.concatenate((X_train,X_critical),axis=0)       #Append critical phase datasets to training set
y_train=np.concatenate((y_train,Y_critical),axis=0)       #Append critical phase labels to training set

y_train=to_categorical(y_train)              #Convert labels to categorical ([0,1] or [1,0])
y_test=to_categorical(y_test)                #for categorical cross entropy
Y_critical=to_categorical(Y_critical)


# %%

"""Define tunable parameters"""

eta=np.logspace(-3,-1,3)                    #Define vector of learning rates (parameter to SGD optimiser)
lamda=0.01                                  #Define hyperparameter
n_layers=1                                  #Define number of hidden layers in the model
n_neuron=np.logspace(0,3,4,dtype=int)       #Define number of neurons per layer
epochs=10                                   #Number of reiterations over the input data
batch_size=100                              #Number of samples per gradient update 

# %%

"""Define function to return Deep Neural Network model"""

def NN_model(inputsize,n_layers,n_neuron,eta,lamda):
    model=Sequential()      
    for i in range(n_layers):       #Run loop to add hidden layers to the model
        if (i==0):                  #First layer requires input dimensions (1600 in this case)
            model.add(Dense(n_neuron,activation='relu',kernel_regularizer=regularizers.l2(lamda),input_dim=inputsize))
        else:                       #Subsequent layers are capable of automatic shape inferencing
            model.add(Dense(n_neuron,activation='relu',kernel_regularizer=regularizers.l2(lamda)))
    model.add(Dense(2,activation='softmax'))  #2 outputs - ordered and disordered (softmax for prob)
    sgd=optimizers.SGD(lr=eta)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model

# %%
    
Train_accuracy=np.zeros((len(n_neuron),len(eta)))      #Define matrices to store accuracy scores as a function
Test_accuracy=np.zeros((len(n_neuron),len(eta)))       #of learning rate and number of hidden neurons for 
Critical_accuracy=np.zeros((len(n_neuron),len(eta)))   #training, testing and critical datasets

for i in range(len(n_neuron)):     #run loops over hidden neurons and learning rates to calculate 
    for j in range(len(eta)):      #accuracy scores 
        DNN_model=NN_model(X_train.shape[1],n_layers,n_neuron[i],eta[j],lamda)
        DNN_model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1)
        Train_accuracy[i,j]=DNN_model.evaluate(X_train,y_train)[1]
        Test_accuracy[i,j]=DNN_model.evaluate(X_test,y_test)[1]
        Critical_accuracy[i,j]=DNN_model.evaluate(X_critical,Y_critical)[1]
        

# %%
        
"""PLot data"""

def plot_data(x,y,data,title=None):

    # plot results
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)
    
    cbar=fig.colorbar(cax)
    cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
    cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
    cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.1f}\\%$".format( 100*data[j,i])  
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y]


    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)

    ax.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{hidden\\ neurons}$',fontsize=fontsize)
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    plt.show()
    
plot_data(eta,n_neuron,Train_accuracy, 'training')
plot_data(eta,n_neuron,Test_accuracy, 'testing')
plot_data(eta,n_neuron,Critical_accuracy, 'critical')
        