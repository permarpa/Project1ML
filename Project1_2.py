# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 13:09:20 2020

@author: Bharat Mishra, Patricia Perez-Martin, Pinelopi Christodoulou
"""

"""Project 1 - Ising Model - Part (c)"""

# %%
"""Import all necessary Python directories"""

import numpy as np
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split as splitter
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.neural_network import MLPClassifier as MLPC

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
plt.title('Ordered Lattice')
plt.show()

plt.figure()                          #Plot one of the critical lattices (any row from 70000 to 99999)
temp=config[70000,:]
critical_config_ex=np.reshape(temp,(L,L))
ax=sns.heatmap(critical_config_ex,vmin=-1.0,vmax=1.0,cmap='plasma_r')
plt.xticks(np.linspace(0,40,9,dtype=int),np.linspace(0,40,9,dtype=int))
plt.yticks(np.linspace(0,40,9,dtype=int),np.linspace(0,40,9,dtype=int))
plt.title('Critical Lattice')
plt.show()

plt.figure()                          #Plot one of the disordered lattices (any row from 100000 to 159999)
temp=config[159999,:]
disorder_config_ex=np.reshape(temp,(L,L))
ax=sns.heatmap(disorder_config_ex,vmin=-1.0,vmax=1.0,cmap='plasma_r')
plt.xticks(np.linspace(0,40,9,dtype=int),np.linspace(0,40,9,dtype=int))
plt.yticks(np.linspace(0,40,9,dtype=int),np.linspace(0,40,9,dtype=int))
plt.title('Disordered Lattice')
plt.show()

del temp

# %%

"""Generate training and testing datasets (no criticality in testing)"""

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

X_train=np.concatenate((X_train,X_critical),axis=0)
y_train=np.concatenate((y_train,Y_critical),axis=0)

# %%

"""Perform logistic regression (various solvers, no criticality) and assign accuracy score"""

lamda=np.logspace(-5,5,11)    #Define array of hyperparameters

train_accuracy_liblin=np.zeros(len(lamda),np.float64)     #Define arrays to store accuracy score for training
test_accuracy_liblin=np.zeros(len(lamda),np.float64)      #testing and critical data under liblinear solver
critical_accuracy_liblin=np.zeros(len(lamda),np.float64)  #(penalty set to default L2)

train_accuracy_liblin_el=np.zeros(len(lamda),np.float64)     #Define arrays to store accuracy score for training
test_accuracy_liblin_el=np.zeros(len(lamda),np.float64)      #testing and critical data under liblinear solver
critical_accuracy_liblin_el=np.zeros(len(lamda),np.float64)  #(penalty changed to L1+L2)

train_accuracy_newt=np.zeros(len(lamda),np.float64)       #Define arrays to store accuracy score for training  
test_accuracy_newt=np.zeros(len(lamda),np.float64)        #testing and critical data under newton-cg solver
critical_accuracy_newt=np.zeros(len(lamda),np.float64)    #(penalty can be only L2)

train_accuracy_sag=np.zeros(len(lamda),np.float64)        #Define arrays to store accuracy score for training
test_accuracy_sag=np.zeros(len(lamda),np.float64)         #testing and critical data under SAG solver
critical_accuracy_sag=np.zeros(len(lamda),np.float64)     #(penalty can be only L2)

train_accuracy_SGD=np.zeros(len(lamda),np.float64)        #Define arrays to store accuracy store for training
test_accuracy_SGD=np.zeros(len(lamda),np.float64)         #testing and critical data under SGD classifier
critical_accuracy_SGD=np.zeros(len(lamda),np.float64)     #(penalty set to default L2)

train_accuracy_SGD_el=np.zeros(len(lamda),np.float64)        #Define arrays to store accuracy store for training
test_accuracy_SGD_el=np.zeros(len(lamda),np.float64)         #testing and critical data under SGD classifier
critical_accuracy_SGD_el=np.zeros(len(lamda),np.float64)     #(penalty changed to L1+L2)

for i in range(len(lamda)):
    
    clf=skl.LogisticRegression(C=1.0/lamda[i],random_state=1,verbose=0,max_iter=1E+3,tol=1E-5,solver='liblinear')
    clf.fit(X_train,y_train)
    train_accuracy_liblin[i]=clf.score(X_train,y_train)                #Log regression using liblinear
    test_accuracy_liblin[i]=clf.score(X_test,y_test)                   #penalty=L2 by default
    critical_accuracy_liblin[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.LogisticRegression(C=1.0/lamda[i],random_state=1,verbose=0,max_iter=1E+3,tol=1E-5,solver='liblinear',penalty='elasticnet')
    clf.fit(X_train,y_train)
    train_accuracy_liblin_el[i]=clf.score(X_train,y_train)                #Log regression using liblinear
    test_accuracy_liblin_el[i]=clf.score(X_test,y_test)                   #penalty=L1+L2
    critical_accuracy_liblin_el[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.LogisticRegression(C=1.0/lamda[i],random_state=1,verbose=0,max_iter=1E+3,tol=1E-5,solver='newton-cg')
    clf.fit(X_train,y_train)
    train_accuracy_newt[i]=clf.score(X_train,y_train)                  #Log regression using Newton-cg
    test_accuracy_newt[i]=clf.score(X_test,y_test)                     #penalty=L2 by default
    critical_accuracy_newt[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.LogisticRegression(C=1.0/lamda[i],random_state=1,verbose=0,max_iter=1E+3,tol=1E-5,solver='sag')
    clf.fit(X_train,y_train)
    train_accuracy_sag[i]=clf.score(X_train,y_train)                  #Log regression using SAG
    test_accuracy_sag[i]=clf.score(X_test,y_test)                     #penalty=L2 by default
    critical_accuracy_sag[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.SGDClassifier(loss='log',penalty='l2',alpha=lamda[i],max_iter=100,shuffle=True,random_state=1,learning_rate='optimal')
    clf.fit(X_train,y_train)
    train_accuracy_SGD[i]=clf.score(X_train,y_train)                  #Log regression using SGD
    test_accuracy_SGD[i]=clf.score(X_test,y_test)                     #penalty=L2 
    critical_accuracy_SGD[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.SGDClassifier(loss='log',penalty='elasticnet',alpha=lamda[i],max_iter=100,shuffle=True,random_state=1,learning_rate='optimal')
    clf.fit(X_train,y_train)
    train_accuracy_SGD_el[i]=clf.score(X_train,y_train)                  #Log regression using SGD
    test_accuracy_SGD_el[i]=clf.score(X_test,y_test)                     #penalty=L1+L2 
    critical_accuracy_SGD_el[i]=clf.score(X_critical,Y_critical)

# %%
    
"""Plot accuracy (no criticality) scores"""

plt.figure()
plt.semilogx(lamda,train_accuracy_liblin,'*-b',label='Training (Liblinear)')
plt.semilogx(lamda,test_accuracy_liblin,'*-r',label='Test (Liblinear)')
plt.semilogx(lamda,critical_accuracy_liblin,'*-g',label='Critical (Liblinear)')
plt.semilogx(lamda,train_accuracy_newt,'--b',label='Training (Newton-CG)')
plt.semilogx(lamda,test_accuracy_newt,'--r',label='Test (Newton-CG)')
plt.semilogx(lamda,critical_accuracy_newt,'--g',label='Critical (Newton-CG)')
plt.semilogx(lamda,train_accuracy_sag,'*--b',label='Training (SAG)')
plt.semilogx(lamda,test_accuracy_sag,'*--r',label='Test (SAG)')
plt.semilogx(lamda,critical_accuracy_sag,'*--g',label='Critical (SAG)')
plt.semilogx(lamda,train_accuracy_SGD,'b',label='Training (SGD)')
plt.semilogx(lamda,test_accuracy_SGD,'r',label='Test (SGD)')
plt.semilogx(lamda,critical_accuracy_SGD,'g',label='Critical (SGD)')
plt.legend()
plt.xlabel('Hyperparameters')
plt.ylabel('Accuracy')
plt.title('Accuracy (Solver Variation, No Criticality)')
plt.show()

plt.figure()
plt.semilogx(lamda,train_accuracy_liblin,'*-b',label='Training (Liblinear-L2)')
plt.semilogx(lamda,test_accuracy_liblin,'*-r',label='Test (Liblinear-L2)')
plt.semilogx(lamda,critical_accuracy_liblin,'*-g',label='Critical (Liblinear-L2)')
plt.semilogx(lamda,train_accuracy_liblin_el,'--b',label='Training (Liblinear-L1+L2)')
plt.semilogx(lamda,test_accuracy_liblin_el,'--r',label='Test (Liblinear-L1+L2)')
plt.semilogx(lamda,critical_accuracy_liblin_el,'--g',label='Critical (Liblinear-L1+L2)')
plt.semilogx(lamda,train_accuracy_SGD,'*--b',label='Training (SGD-L2)')
plt.semilogx(lamda,test_accuracy_SGD,'*--r',label='Test (SGD-L2)')
plt.semilogx(lamda,critical_accuracy_SGD,*--'g',label='Critical (SGD-L2)')
plt.semilogx(lamda,train_accuracy_SGD_el,'b',label='Training (SGD-L1+L2)')
plt.semilogx(lamda,test_accuracy_SGD_el,'r',label='Test (SGD-L1+L2)')
plt.semilogx(lamda,critical_accuracy_SGD_el,'g',label='Critical (SGD-L1+L2)')
plt.xlabel('Hyperparameters')
plt.ylabel('Accuracy')
plt.title('Accuracy (Penalty Variation, No Criticality)')
plt.legend()
plt.show()

# %%

"""Generate training and testing datasets (include criticality in testing)"""

X_ordered=config[:70000,:]         #Collect all ordered spin configurations
Y_ordered=label[:70000,0]          #Collect all ordered labels

X_critical=config[70000:100000,:]    #Collect all critical spin configurations
Y_critical=label[70000:100000,0]     #Collect all critical labels

X_disordered=config[100000:,:]      #Collect all disorderd spin configurations
Y_disordered=label[100000:,0]       #Collect all disordered labels

X_train,X_test,y_train,y_test=splitter(config,label,test_size=0.5)


# %%

"""Perform logistic regression (various solvers, with criticality) and assign accuracy score"""

train_accuracy_liblin_cr=np.zeros(len(lamda),np.float64)     #Define arrays to store accuracy score for training
test_accuracy_liblin_cr=np.zeros(len(lamda),np.float64)      #testing and critical data under liblinear solver
critical_accuracy_liblin_cr=np.zeros(len(lamda),np.float64)  #(penalty set to default L2)

train_accuracy_liblin_el_cr=np.zeros(len(lamda),np.float64)     #Define arrays to store accuracy score for training
test_accuracy_liblin_el_cr=np.zeros(len(lamda),np.float64)      #testing and critical data under liblinear solver
critical_accuracy_liblin_el_cr=np.zeros(len(lamda),np.float64)  #(penalty changed to L1+L2)

train_accuracy_newt_cr=np.zeros(len(lamda),np.float64)       #Define arrays to store accuracy score for training  
test_accuracy_newt_cr=np.zeros(len(lamda),np.float64)        #testing and critical data under newton-cg solver
critical_accuracy_newt_cr=np.zeros(len(lamda),np.float64)    #(penalty can be only L2)

train_accuracy_sag_cr=np.zeros(len(lamda),np.float64)        #Define arrays to store accuracy score for training
test_accuracy_sag_cr=np.zeros(len(lamda),np.float64)         #testing and critical data under SAG solver
critical_accuracy_sag_cr=np.zeros(len(lamda),np.float64)     #(penalty can be only L2)

train_accuracy_SGD_cr=np.zeros(len(lamda),np.float64)        #Define arrays to store accuracy store for training
test_accuracy_SGD_cr=np.zeros(len(lamda),np.float64)         #testing and critical data under SGD classifier
critical_accuracy_SGD_cr=np.zeros(len(lamda),np.float64)     #(penalty set to default L2)

train_accuracy_SGD_el_cr=np.zeros(len(lamda),np.float64)        #Define arrays to store accuracy store for training
test_accuracy_SGD_el_cr=np.zeros(len(lamda),np.float64)         #testing and critical data under SGD classifier
critical_accuracy_SGD_el_cr=np.zeros(len(lamda),np.float64)     #(penalty changed to L1+L2)

for i in range(len(lamda)):
    
    clf=skl.LogisticRegression(C=1.0/lamda[i],random_state=1,verbose=0,max_iter=1E+3,tol=1E-5,solver='liblinear')
    clf.fit(X_train,y_train)
    train_accuracy_liblin_cr[i]=clf.score(X_train,y_train)                #Log regression using liblinear
    test_accuracy_liblin_cr[i]=clf.score(X_test,y_test)                   #penalty=L2 by default
    critical_accuracy_liblin_cr[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.LogisticRegression(C=1.0/lamda[i],random_state=1,verbose=0,max_iter=1E+3,tol=1E-5,solver='liblinear',penalty='elasticnet')
    clf.fit(X_train,y_train)
    train_accuracy_liblin_el_cr[i]=clf.score(X_train,y_train)                #Log regression using liblinear
    test_accuracy_liblin_el_cr[i]=clf.score(X_test,y_test)                   #penalty=L1+L2
    critical_accuracy_liblin_el_cr[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.LogisticRegression(C=1.0/lamda[i],random_state=1,verbose=0,max_iter=1E+3,tol=1E-5,solver='newton-cg')
    clf.fit(X_train,y_train)
    train_accuracy_newt_cr[i]=clf.score(X_train,y_train)                  #Log regression using Newton-cg
    test_accuracy_newt_cr[i]=clf.score(X_test,y_test)                     #penalty=L2 by default
    critical_accuracy_newt_cr[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.LogisticRegression(C=1.0/lamda[i],random_state=1,verbose=0,max_iter=1E+3,tol=1E-5,solver='sag')
    clf.fit(X_train,y_train)
    train_accuracy_sag_cr[i]=clf.score(X_train,y_train)                  #Log regression using SAG
    test_accuracy_sag_cr[i]=clf.score(X_test,y_test)                     #penalty=L2 by default
    critical_accuracy_sag_cr[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.SGDClassifier(loss='log',penalty='l2',alpha=lamda[i],max_iter=100,shuffle=True,random_state=1,learning_rate='optimal')
    clf.fit(X_train,y_train)
    train_accuracy_SGD_cr[i]=clf.score(X_train,y_train)                  #Log regression using SGD
    test_accuracy_SGD_cr[i]=clf.score(X_test,y_test)                     #penalty=L2 
    critical_accuracy_SGD_cr[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.SGDClassifier(loss='log',penalty='elasticnet',alpha=lamda[i],max_iter=100,shuffle=True,random_state=1,learning_rate='optimal')
    clf.fit(X_train,y_train)
    train_accuracy_SGD_el_cr[i]=clf.score(X_train,y_train)                  #Log regression using SGD
    test_accuracy_SGD_el_cr[i]=clf.score(X_test,y_test)                     #penalty=L1+L2 
    critical_accuracy_SGD_el_cr[i]=clf.score(X_critical,Y_critical)

# %%
    
"""Plot accuracy (criticality) scores"""

plt.figure()
plt.semilogx(lamda,train_accuracy_liblin_cr,'*-b',label='Training (Liblinear)')
plt.semilogx(lamda,test_accuracy_liblin_cr,'*-r',label='Test (Liblinear)')
plt.semilogx(lamda,critical_accuracy_liblin_cr,'*-g',label='Critical (Liblinear)')
plt.semilogx(lamda,train_accuracy_newt_cr,'--b',label='Training (Newton-CG)')
plt.semilogx(lamda,test_accuracy_newt_cr,'--r',label='Test (Newton-CG)')
plt.semilogx(lamda,critical_accuracy_newt_cr,'--g',label='Critical (Newton-CG)')
plt.semilogx(lamda,train_accuracy_sag_cr,'*--b',label='Training (SAG)')
plt.semilogx(lamda,test_accuracy_sag_cr,'*--r',label='Test (SAG)')
plt.semilogx(lamda,critical_accuracy_sag_cr,'*--g',label='Critical (SAG)')
plt.semilogx(lamda,train_accuracy_SGD_cr,'b',label='Training (SGD)')
plt.semilogx(lamda,test_accuracy_SGD_cr,'r',label='Test (SGD)')
plt.semilogx(lamda,critical_accuracy_SGD_cr,'g',label='Critical (SGD)')
plt.xlabel('Hyperparameters')
plt.ylabel('Accuracy')
plt.title('Accuracy (Solver Variation, Criticality)')
plt.legend()
plt.show()

plt.figure()
plt.semilogx(lamda,train_accuracy_liblin_cr,'*-b',label='Training (Liblinear-L2)')
plt.semilogx(lamda,test_accuracy_liblin_cr,'*-r',label='Test (Liblinear-L2)')
plt.semilogx(lamda,critical_accuracy_liblin_cr,'*-g',label='Critical (Liblinear-L2)')
plt.semilogx(lamda,train_accuracy_liblin_el_cr,'--b',label='Training (Liblinear-L1+L2)')
plt.semilogx(lamda,test_accuracy_liblin_el_cr,'--r',label='Test (Liblinear-L1+L2)')
plt.semilogx(lamda,critical_accuracy_liblin_el_cr,'--g',label='Critical (Liblinear-L1+L2)')
plt.semilogx(lamda,train_accuracy_SGD_cr,'*--b',label='Training (SGD-L2)')
plt.semilogx(lamda,test_accuracy_SGD_cr,'*--r',label='Test (SGD-L2)')
plt.semilogx(lamda,critical_accuracy_SGD_cr,*--'g',label='Critical (SGD-L2)')
plt.semilogx(lamda,train_accuracy_SGD_el_cr,'b',label='Training (SGD-L1+L2)')
plt.semilogx(lamda,test_accuracy_SGD_el_cr,'r',label='Test (SGD-L1+L2)')
plt.semilogx(lamda,critical_accuracy_SGD_el_cr,'g',label='Critical (SGD-L1+L2)')
plt.xlabel('Hyperparameters')
plt.ylabel('Accuracy')
plt.title('Accuracy (Penalty Variation, Criticality)')
plt.legend()
plt.show()

    

    
    







