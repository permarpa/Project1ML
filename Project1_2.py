# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 13:09:20 2020

@author: Bharat Mishra, Patricia Perez-Martin, Pinelopi Christodoulou
"""

"""Project 1 - Ising Model - Part (c)"""

# %%
"""Import all necessary Python libraries"""

import numpy as np
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split as splitter
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import KFold

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
plt.title('Ordered Phase',fontweight='bold')
plt.show()

plt.figure()                          #Plot one of the critical lattices (any row from 70000 to 99999)
temp=config[70000,:]
critical_config_ex=np.reshape(temp,(L,L))
ax=sns.heatmap(critical_config_ex,vmin=-1.0,vmax=1.0,cmap='plasma_r')
plt.xticks(np.linspace(0,40,9,dtype=int),np.linspace(0,40,9,dtype=int))
plt.yticks(np.linspace(0,40,9,dtype=int),np.linspace(0,40,9,dtype=int))
plt.title('Critical Phase',fontweight='bold')
plt.show()

plt.figure()                          #Plot one of the disordered lattices (any row from 100000 to 159999)
temp=config[159999,:]
disorder_config_ex=np.reshape(temp,(L,L))
ax=sns.heatmap(disorder_config_ex,vmin=-1.0,vmax=1.0,cmap='plasma_r')
plt.xticks(np.linspace(0,40,9,dtype=int),np.linspace(0,40,9,dtype=int))
plt.yticks(np.linspace(0,40,9,dtype=int),np.linspace(0,40,9,dtype=int))
plt.title('Disordered Phase',fontweight='bold')
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

X_train=np.concatenate((X_train,X_critical),axis=0)      #Append critical phase datasets to training data
y_train=np.concatenate((y_train,Y_critical),axis=0)      #Append critical phase labels to training data

# %%

"""Perform logistic regression (various solvers and penalties, no criticality) and assign accuracy score"""
#Using different solvers (liblinear, SAGA, SGDClassifier) and different regularisation schemes (L2 and L1+L2)

lamda=np.logspace(-5,5,11)    #Define array of hyperparameters

train_accuracy_liblin=np.zeros(len(lamda),np.float64)     #Define arrays to store accuracy score for training
test_accuracy_liblin=np.zeros(len(lamda),np.float64)      #testing and critical data under liblinear solver
critical_accuracy_liblin=np.zeros(len(lamda),np.float64)  #(penalty set to default L2)

train_accuracy_saga_el=np.zeros(len(lamda),np.float64)     #Define arrays to store accuracy score for training
test_accuracy_saga_el=np.zeros(len(lamda),np.float64)      #testing and critical data under SAGA solver
critical_accuracy_saga_el=np.zeros(len(lamda),np.float64)  #(penalty changed to L1+L2)

train_accuracy_saga=np.zeros(len(lamda),np.float64)        #Define arrays to store accuracy score for training
test_accuracy_saga=np.zeros(len(lamda),np.float64)         #testing and critical data under SAGA solver
critical_accuracy_saga=np.zeros(len(lamda),np.float64)     #(penalty set to default L2)

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
    
    clf=skl.LogisticRegression(C=1.0/lamda[i],random_state=1,verbose=0,max_iter=1E+3,tol=1E-5,solver='saga',penalty='elasticnet',l1_ratio=0.3)
    clf.fit(X_train,y_train)
    train_accuracy_saga_el[i]=clf.score(X_train,y_train)                #Log regression usig SAGA
    test_accuracy_saga_el[i]=clf.score(X_test,y_test)                   #penalty=L1+L2
    critical_accuracy_saga_el[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.LogisticRegression(C=1.0/lamda[i],random_state=1,verbose=0,max_iter=1E+3,tol=1E-5,solver='saga')
    clf.fit(X_train,y_train)
    train_accuracy_saga[i]=clf.score(X_train,y_train)                  #Log regression using SAGA
    test_accuracy_saga[i]=clf.score(X_test,y_test)                     #penalty=L2 by default
    critical_accuracy_saga[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.SGDClassifier(loss='log',penalty='l2',alpha=lamda[i],max_iter=1E+3,shuffle=True,random_state=1,learning_rate='optimal')
    clf.fit(X_train,y_train)
    train_accuracy_SGD[i]=clf.score(X_train,y_train)                  #Log regression using SGD
    test_accuracy_SGD[i]=clf.score(X_test,y_test)                     #penalty=L2 
    critical_accuracy_SGD[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.SGDClassifier(loss='log',penalty='elasticnet',alpha=lamda[i],max_iter=1E+3,shuffle=True,random_state=1,learning_rate='optimal',l1_ratio=0.3)
    clf.fit(X_train,y_train)
    train_accuracy_SGD_el[i]=clf.score(X_train,y_train)                  #Log regression using SGD
    test_accuracy_SGD_el[i]=clf.score(X_test,y_test)                     #penalty=L1+L2 
    critical_accuracy_SGD_el[i]=clf.score(X_critical,Y_critical)

# %%
    
"""Plot accuracy (no criticality) scores"""

plt.figure()
plt.semilogx(lamda,train_accuracy_liblin,'*-b',linewidth=1.5,label='Training (Liblinear)')
plt.semilogx(lamda,test_accuracy_liblin,'*-r',linewidth=1.5,label='Test (Liblinear)')
plt.semilogx(lamda,critical_accuracy_liblin,'*-g',linewidth=1.5,label='Critical (Liblinear)')
plt.semilogx(lamda,train_accuracy_saga,'*--b',linewidth=1.5,label='Training (SAGA)')
plt.semilogx(lamda,test_accuracy_saga,'*--r',linewidth=1.5,label='Test (SAGA)')
plt.semilogx(lamda,critical_accuracy_saga,'*--g',linewidth=1.5,label='Critical (SAGA)')
plt.semilogx(lamda,train_accuracy_SGD,'b',linewidth=1.5,label='Training (SGD)')
plt.semilogx(lamda,test_accuracy_SGD,'r',linewidth=1.5,label='Test (SGD)')
plt.semilogx(lamda,critical_accuracy_SGD,'g',linewidth=1.5,label='Critical (SGD)')
plt.legend()
plt.grid()
plt.xlabel('\u03BB',fontweight='bold')
plt.ylabel('Accuracy',fontweight='bold')
plt.title('System Performance (Solver Variation, No Criticality)',fontweight='bold')
plt.show()

plt.figure()
plt.semilogx(lamda,train_accuracy_saga,'*-b',linewidth=1.5,label='Training (SAGA-L2)')
plt.semilogx(lamda,test_accuracy_saga,'*-r',linewidth=1.5,label='Test (SAGA-L2)')
plt.semilogx(lamda,critical_accuracy_saga,'*-g',linewidth=1.5,label='Critical (SAGA-L2)')
plt.semilogx(lamda,train_accuracy_saga_el,'--b',linewidth=1.5,label='Training (SAGA-L1+L2)')
plt.semilogx(lamda,test_accuracy_saga_el,'--r',linewidth=1.5,label='Test (SAGA-L1+L2)')
plt.semilogx(lamda,critical_accuracy_saga_el,'--g',label='Critical (SAGA-L1+L2)')
plt.semilogx(lamda,train_accuracy_SGD,'*--b',linewidth=1.5,label='Training (SGD-L2)')
plt.semilogx(lamda,test_accuracy_SGD,'*--r',linewidth=1.5,label='Test (SGD-L2)')
plt.semilogx(lamda,critical_accuracy_SGD,*--'g',linewidth=1.5,label='Critical (SGD-L2)')
plt.semilogx(lamda,train_accuracy_SGD_el,'b',linewidth=1.5,label='Training (SGD-L1+L2)')
plt.semilogx(lamda,test_accuracy_SGD_el,'r',linewidth=1.5,label='Test (SGD-L1+L2)')
plt.semilogx(lamda,critical_accuracy_SGD_el,'g',linewidth=1.5,label='Critical (SGD-L1+L2)')
plt.legend()
plt.grid()
plt.xlabel('\u03BB',fontweight='bold')
plt.ylabel('Accuracy',fontweight='bold')
plt.title('System Performance (Penalty Variation, No Criticality)',fontweight='bold')
plt.show()

# %%

"""Generate training and testing datasets (include criticality in testing)"""

X_ordered=config[:70000,:]         #Collect all ordered spin configurations
Y_ordered=label[:70000,0]          #Collect all ordered labels

X_critical=config[70000:100000,:]    #Collect all critical spin configurations
Y_critical=label[70000:100000,0]     #Collect all critical labels

X_disordered=config[100000:,:]      #Collect all disorderd spin configurations
Y_disordered=label[100000:,0]       #Collect all disordered labels

X_train,X_test,y_train,y_test=splitter(config,label,test_size=0.5)    #Direct splitting of config to include
                                                                      #critical phase matrices in test


# %%

"""Perform logistic regression (various solvers, with criticality) and assign accuracy score"""
#With criticality means the splitting into training and testing data is performed for all phases

train_accuracy_liblin_cr=np.zeros(len(lamda),np.float64)     #Define arrays to store accuracy score for training
test_accuracy_liblin_cr=np.zeros(len(lamda),np.float64)      #testing and critical data under liblinear solver
critical_accuracy_liblin_cr=np.zeros(len(lamda),np.float64)  #(penalty set to default L2)

train_accuracy_saga_el_cr=np.zeros(len(lamda),np.float64)     #Define arrays to store accuracy score for training
test_accuracy_saga_el_cr=np.zeros(len(lamda),np.float64)      #testing and critical data under SAGA solver
critical_accuracy_saga_el_cr=np.zeros(len(lamda),np.float64)  #(penalty changed to L1+L2)

train_accuracy_saga_cr=np.zeros(len(lamda),np.float64)        #Define arrays to store accuracy score for training
test_accuracy_saga_cr=np.zeros(len(lamda),np.float64)         #testing and critical data under SAGA solver
critical_accuracy_saga_cr=np.zeros(len(lamda),np.float64)     #(penalty set to default L2)

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
    
    clf=skl.LogisticRegression(C=1.0/lamda[i],random_state=1,verbose=0,max_iter=1E+3,tol=1E-5,solver='saga',penalty='elasticnet',l1_ratio=0.3)
    clf.fit(X_train,y_train)
    train_accuracy_saga_el_cr[i]=clf.score(X_train,y_train)                #Log regression using SAGA
    test_accuracy_saga_el_cr[i]=clf.score(X_test,y_test)                   #penalty=L1+L2
    critical_accuracy_saga_el_cr[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.LogisticRegression(C=1.0/lamda[i],random_state=1,verbose=0,max_iter=1E+3,tol=1E-5,solver='saga')
    clf.fit(X_train,y_train)
    train_accuracy_saga_cr[i]=clf.score(X_train,y_train)                  #Log regression using SAGA
    test_accuracy_saga_cr[i]=clf.score(X_test,y_test)                     #penalty=L2 by default
    critical_accuracy_saga_cr[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.SGDClassifier(loss='log',penalty='l2',alpha=lamda[i],max_iter=1E+3,shuffle=True,random_state=1,learning_rate='optimal')
    clf.fit(X_train,y_train)
    train_accuracy_SGD_cr[i]=clf.score(X_train,y_train)                  #Log regression using SGD
    test_accuracy_SGD_cr[i]=clf.score(X_test,y_test)                     #penalty=L2 
    critical_accuracy_SGD_cr[i]=clf.score(X_critical,Y_critical)
    
    clf=skl.SGDClassifier(loss='log',penalty='elasticnet',alpha=lamda[i],max_iter=1E+3,shuffle=True,random_state=1,learning_rate='optimal',l1_ratio=0.3)
    clf.fit(X_train,y_train)
    train_accuracy_SGD_el_cr[i]=clf.score(X_train,y_train)                  #Log regression using SGD
    test_accuracy_SGD_el_cr[i]=clf.score(X_test,y_test)                     #penalty=L1+L2 
    critical_accuracy_SGD_el_cr[i]=clf.score(X_critical,Y_critical)

# %%
    
"""Plot accuracy (criticality) scores"""

plt.figure()
plt.semilogx(lamda,train_accuracy_liblin_cr,'*-b',linewidth=1.5,label='Training (Liblinear)')
plt.semilogx(lamda,test_accuracy_liblin_cr,'*-r',linewidth=1.5,label='Test (Liblinear)')
plt.semilogx(lamda,critical_accuracy_liblin_cr,'*-g',linewidth=1.5,label='Critical (Liblinear)')
plt.semilogx(lamda,train_accuracy_saga_cr,'*--b',linewidth=1.5,label='Training (SAGA)')
plt.semilogx(lamda,test_accuracy_saga_cr,'*--r',linewidth=1.5,label='Test (SAGA)')
plt.semilogx(lamda,critical_accuracy_saga_cr,'*--g',linewidth=1.5,label='Critical (SAGA)')
plt.semilogx(lamda,train_accuracy_SGD_cr,'b',linewidth=1.5,label='Training (SGD)')
plt.semilogx(lamda,test_accuracy_SGD_cr,'r',linewidth=1.5,label='Test (SGD)')
plt.semilogx(lamda,critical_accuracy_SGD_cr,'g',linewidth=1.5,label='Critical (SGD)')
plt.legend()
plt.grid()
plt.xlabel('\u03BB',fontweight='bold')
plt.ylabel('Accuracy',fontweight='bold')
plt.title('System Performance (Solver Variation, With Criticality)',fontweight='bold')
plt.show()

plt.figure()
plt.semilogx(lamda,train_accuracy_saga_cr,'*-b',linewidth=1.5,label='Training (SAGA-L2)')
plt.semilogx(lamda,test_accuracy_saga_cr,'*-r',linewidth=1.5,label='Test (SAGA-L2)')
plt.semilogx(lamda,critical_accuracy_saga_cr,'*-g',linewidth=1.5,label='Critical (SAGA-L2)')
plt.semilogx(lamda,train_accuracy_saga_el_cr,'--b',linewidth=1.5,label='Training (SAGA-L1+L2)')
plt.semilogx(lamda,test_accuracy_saga_el_cr,'--r',linewidth=1.5,label='Test (SAGA-L1+L2)')
plt.semilogx(lamda,critical_accuracy_saga_el_cr,'--g',linewidth=1.5,label='Critical (SAGA-L1+L2)')
plt.semilogx(lamda,train_accuracy_SGD_cr,'*--b',linewidth=1.5,label='Training (SGD-L2)')
plt.semilogx(lamda,test_accuracy_SGD_cr,'*--r',linewidth=1.5,label='Test (SGD-L2)')
plt.semilogx(lamda,critical_accuracy_SGD_cr,*--'g',linewidth=1.5,label='Critical (SGD-L2)')
plt.semilogx(lamda,train_accuracy_SGD_el_cr,'b',linewidth=1.5,label='Training (SGD-L1+L2)')
plt.semilogx(lamda,test_accuracy_SGD_el_cr,'r',linewidth=1.5,label='Test (SGD-L1+L2)')
plt.semilogx(lamda,critical_accuracy_SGD_el_cr,'g',linewidth=1.5,label='Critical (SGD-L1+L2)')
plt.legend()
plt.grid()
plt.xlabel('\u03BB',fontweight='bold')
plt.ylabel('Accuracy',fontweight='bold')
plt.title('System Performance (Penalty Variation, With Criticality)',fontweight='bold')
plt.show()

# %%

"""Implement cross-validation framework (only on liblinear solver and no criticality)"""

#Initiate k-fold instance for implementing manual cross-validation using KFold

k=5
kfold=KFold(n_splits=k)

train_scores=np.zeros((len(lamda),k))
test_scores=np.zeros((len(lamda),k))
critical_scores=np.zeros((len(lamda),k))

X_temp=np.concatenate((X_ordered,X_disordered),axis=0)    #Combine ordered and disordered spins for split
Y_temp=np.concatenate((Y_ordered,Y_disordered),axis=0)    #Combine ordered and disordered labels for split

for i in range(len(lamda)):
    j=0
    for train_inds,test_inds in kfold.split(X_temp):    #Split data into training and testing
        X_train=X_temp[train_inds]
        y_train=Y_temp[train_inds]
        X_test=X_temp[test_inds]
        y_test=Y_temp[test_inds]
        
        X_train=np.concatenate((X_train,X_critical),axis=0)   #Append critical phase configurations to train 
        y_train=np.concatenate((y_train,Y_critical),axis=0)   #Append critical phase labels to train
        
        clf=skl.LogisticRegression(C=1.0/lamda[i],random_state=1,verbose=0,max_iter=1E+3,tol=1E-5,solver='liblinear')
        clf.fit(X_train,y_train)
        train_scores[i,j]=clf.score(X_train,y_train)
        test_scores[i,j]=clf.score(X_test,y_test)
        critical_scores[i,j]=clf.score(X_critical,Y_critical)
        j+=1
        
train_accuracy_liblin_cv_kfold=np.mean(train_scores,axis=1)
test_accuracy_liblin_cv_kfold=np.mean(test_scores,axis=1)
critical_accuracy_liblin_cv_kfold=np.mean(critical_scores,axis=1)

del X_temp,Y_temp
# %%
    
"""Plot accuracy (no criticality) with and without cross-validation"""

plt.figure()
plt.semilogx(lamda,train_accuracy_liblin,'*-b',linewidth=1.5,label='Training (Liblinear)')
plt.semilogx(lamda,test_accuracy_liblin,'*-r',linewidth=1.5,label='Test (Liblinear)')
plt.semilogx(lamda,critical_accuracy_liblin,'*-g',linewidth=1.5,label='Critical (Liblinear)')
plt.semilogx(lamda,train_accuracy_liblin_cv_kfold,'--b',linewidth=1.5,label='Training (Liblinear+CV)')
plt.semilogx(lamda,test_accuracy_liblin_cv_kfold,'--r',linewidth=1.5,label='Test (Liblinear+CV)')
plt.semilogx(lamda,critical_accuracy_liblin_cv_kfold,'--g',linewidth=1.5,label='Critical (Liblinear+CV)')
plt.legend()
plt.grid()
plt.xlabel('\u03BB',fontweight='bold')
plt.ylabel('Accuracy',fontweight='bold')
plt.title('System Performance (5-fold Cross Validation)')
plt.show()


    

    
    







