# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:24:47 2020

@authors: Bharat Mishra, Patricia Perez-Martin, Pinelopi Christodoulou
"""

"""Project 1 - Ising Model - Part (a)+(b)"""

# %%
"""Import all necessary Python directories"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error as MSE
#from sklearn.metrics import r2_score as R2
from sklearn.model_selection import train_test_split as splitter
from sklearn.utils import resample
import seaborn as sns

# %%
"""Part (a) - Create dataset"""

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

# %%

"""Part (b) - Perform regression"""

#Define function to create design matrix

def Design_Matrix(state):              #Design matrix consists of all 2 spin interactions
    L=state.shape[1]
    n=state.shape[0]
    X=np.zeros((n,L**2))
    for i in range(X.shape[0]):
        j=0
        for k in range(L):
            for l in range(L):
                X[i,j]=int(state[i,k]*state[i,l])
                j+=1
    Xmean=np.ones(n)
    Xmean=np.reshape(Xmean,(n,1))
    X=np.hstack((Xmean,X))
    return X

#Define function to calculate the learned interaction
    
def Learned_Interaction(coeff):
    IntMat=np.reshape(coeff[1:],(40,40))
    return IntMat
     
#Construct the design matrix
                    
X=Design_Matrix(state)

#Split the data into training and test

X_train,X_test,y_train,y_test=splitter(X,Energy,test_size=0.96)

#Define vector of hyperparameter values for OLS, Ridge and Lasso and perform regression

lamda=np.logspace(-4,4,9)

MSE_OLS_train=np.zeros(len(lamda))
MSE_Ridge_train=np.zeros(len(lamda))
MSE_Lasso_train=np.zeros(len(lamda))
R2_OLS_train=np.zeros(len(lamda))       #Define vectors to store MSE and R2 values
R2_Ridge_train=np.zeros(len(lamda))     #For training and testing data
R2_Lasso_train=np.zeros(len(lamda))

MSE_OLS_test=np.zeros(len(lamda))
MSE_Ridge_test=np.zeros(len(lamda))
MSE_Lasso_test=np.zeros(len(lamda))
R2_OLS_test=np.zeros(len(lamda))
R2_Ridge_test=np.zeros(len(lamda))
R2_Lasso_test=np.zeros(len(lamda))

#Define function to calculate R2 values

def R2(ypred,ytrue):
    term1=np.sum((ytrue-ypred)**2)
    term2=np.mean(ypred)
    term3=np.sum((ytrue-term2)**2)
    R2score=1-(term1/term3)
    return R2score

for i in range(len(lamda)):
    clf=skl.LinearRegression(fit_intercept=False).fit(X_train,y_train)
    ypred_train=clf.predict(X_train)
    if (i==4):
        OLS_IntMat=Learned_Interaction(clf.coef_)
    ypred_test=clf.predict(X_test)
    MSE_OLS_train[i]=MSE(ypred_train,y_train)
    MSE_OLS_test[i]=MSE(ypred_test,y_test)
    R2_OLS_train[i]=R2(ypred_train,y_train)
    R2_OLS_test[i]=R2(ypred_test,y_test)
    
    clf=skl.Ridge(alpha=lamda[i],fit_intercept=False).fit(X_train,y_train)
    ypred_train=clf.predict(X_train)
    if (i==4):
        Ridge_IntMat=Learned_Interaction(clf.coef_)
    ypred_test=clf.predict(X_test)
    MSE_Ridge_train[i]=MSE(ypred_train,y_train)
    MSE_Ridge_test[i]=MSE(ypred_test,y_test)
    R2_Ridge_train[i]=R2(ypred_train,y_train)
    R2_Ridge_test[i]=R2(ypred_test,y_test)
    
    clf=skl.Lasso(alpha=lamda[i],fit_intercept=False).fit(X_train,y_train)
    ypred_train=clf.predict(X_train)
    if (i==4):
        Lasso_IntMat=Learned_Interaction(clf.coef_)
    ypred_test=clf.predict(X_test)
    MSE_Lasso_train[i]=MSE(ypred_train,y_train)
    MSE_Lasso_test[i]=MSE(ypred_test,y_test)
    R2_Lasso_train[i]=R2(ypred_train,y_train)
    R2_Lasso_test[i]=R2(ypred_test,y_test)
    

#%%%

"""Plot unsampled results"""

plt.figure()
plt.semilogx(lamda,MSE_OLS_train,'b',label='Train (OLS)')
plt.semilogx(lamda,MSE_OLS_test,'--b',label='Test (OLS)')
plt.semilogx(lamda,MSE_Ridge_train,'r',label='Train (Ridge)')
plt.semilogx(lamda,MSE_Ridge_test,'--r',label='Test (Ridge)')
plt.semilogx(lamda,MSE_Lasso_train,'g',label='Train (Lasso)')
plt.semilogx(lamda,MSE_Lasso_test,'--g',label='Test (Lasso)')
plt.legend()
plt.xlabel('Hyperparameters')
plt.ylabel('MSE')
plt.title('MSE')
plt.show()

plt.figure()
plt.semilogx(lamda,R2_OLS_train,'b',label='Train (OLS)')
plt.semilogx(lamda,R2_OLS_test,'--b',label='Test (OLS)')
plt.semilogx(lamda,R2_Ridge_train,'r',label='Train (Ridge)')
plt.semilogx(lamda,R2_Ridge_test,'--r',label='Test (Ridge)')
plt.semilogx(lamda,R2_Lasso_train,'g',label='Train (Lasso)')
plt.semilogx(lamda,R2_Lasso_test,'--g',label='Test (Lasso)')
plt.legend()
plt.xlabel('Hyperparameters')
plt.ylabel('R2')
plt.title('R2')
plt.show()

plt.figure()
ax=sns.heatmap(OLS_IntMat,vmin=-1.0,vmax=1.0,cmap='seismic')
plt.title('Lamda=1e-4, OLS')
plt.show()

plt.figure()
ax=sns.heatmap(Ridge_IntMat,vmin=-1.0,vmax=1.0,cmap='seismic')
plt.title('Lamda=1e-4, Ridge')
plt.show()
                    
plt.figure()
ax=sns.heatmap(Lasso_IntMat,vmin=-1.0,vmax=1.0,cmap='seismic')
plt.title('Lamda=1e-4, Lasso')
plt.show()

# %%

"""Apply Bootstrap framework"""

#Define Bootstrap parameters

n_bootstrap=500
MSE_boot_OLS_train=np.zeros(len(lamda))
MSE_boot_OLS_test=np.zeros(len(lamda))
MSE_boot_Ridge_train=np.zeros(len(lamda))
MSE_boot_Ridge_test=np.zeros(len(lamda))
MSE_boot_Lasso_train=np.zeros(len(lamda))
MSE_boot_Lasso_test=np.zeros(len(lamda))
Var_OLS_train=np.zeros(len(lamda))
Var_OLS_test=np.zeros(len(lamda))
Var_Ridge_train=np.zeros(len(lamda))
Var_Ridge_test=np.zeros(len(lamda))
Var_Lasso_train=np.zeros(len(lamda))
Var_Lasso_test=np.zeros(len(lamda))
Bias_OLS_train=np.zeros(len(lamda))
Bias_OLS_test=np.zeros(len(lamda))
Bias_Ridge_train=np.zeros(len(lamda))
Bias_Ridge_test=np.zeros(len(lamda))
Bias_Lasso_train=np.zeros(len(lamda))
Bias_Lasso_test=np.zeros(len(lamda))

y_train=np.reshape(y_train,(len(y_train),1))
y_test=np.reshape(y_test,(len(y_test),1))
ypred_train_OLS=np.zeros((len(y_train),n_bootstrap))
ypred_test_OLS=np.zeros((len(y_test),n_bootstrap))
ypred_train_Ridge=np.zeros((len(y_train),n_bootstrap))
ypred_test_Ridge=np.zeros((len(y_test),n_bootstrap))
ypred_train_Lasso=np.zeros((len(y_train),n_bootstrap))
ypred_test_Lasso=np.zeros((len(y_test),n_bootstrap))

for i in range(len(lamda)):
    for j in range(n_bootstrap):
        X_,y_=resample(X_train,y_train)
        clf=skl.LinearRegression(fit_intercept=False).fit(X_,y_)
        ypred_train_OLS[:,j]=clf.predict(X_).flatten()
        ypred_test_OLS[:,j]=clf.predict(X_test).flatten()
        
        clf=skl.Ridge(alpha=lamda[i],fit_intercept=False).fit(X_,y_)
        ypred_train_Ridge[:,j]=clf.predict(X_).flatten()
        ypred_test_Ridge[:,j]=clf.predict(X_test).flatten()
        
        clf=skl.Lasso(alpha=lamda[i],fit_intercept=False).fit(X_,y_)
        ypred_train_Lasso[:,j]=clf.predict(X_).flatten()
        ypred_test_Lasso[:,j]=clf.predict(X_test).flatten()
        
    MSE_boot_OLS_train[i]=np.mean(np.mean((y_train-ypred_train_OLS)**2,axis=1,keepdims=True))
    MSE_boot_OLS_test[i]=np.mean(np.mean((y_test-ypred_test_OLS)**2,axis=1,keepdims=True))
    MSE_boot_Ridge_train[i]=np.mean(np.mean((y_train-ypred_train_Ridge)**2,axis=1,keepdims=True))
    MSE_boot_Ridge_test[i]=np.mean(np.mean((y_test-ypred_test_Ridge)**2,axis=1,keepdims=True))
    MSE_boot_Lasso_train[i]=np.mean(np.mean((y_train-ypred_train_Lasso)**2,axis=1,keepdims=True))
    MSE_boot_Lasso_test[i]=np.mean(np.mean((y_test-ypred_test_Lasso)**2,axis=1,keepdims=True))
    
    Var_OLS_train[i]=np.mean(np.var(ypred_train_OLS,axis=1,keepdims=True))
    Var_OLS_test[i]=np.mean(np.var(ypred_test_OLS,axis=1,keepdims=True))
    Var_Ridge_train[i]=np.mean(np.var(ypred_train_Ridge,axis=1,keepdims=True))
    Var_Ridge_test[i]=np.mean(np.var(ypred_test_Ridge,axis=1,keepdims=True))
    Var_Lasso_train[i]=np.mean(np.var(ypred_train_Lasso,axis=1,keepdims=True))
    Var_Lasso_test[i]=np.mean(np.var(ypred_test_Lasso,axis=1,keepdims=True))
    
    Bias_OLS_train[i]=np.mean((y_train-np.mean(ypred_train_OLS,axis=1,keepdims=True))**2)
    Bias_OLS_test[i]=np.mean((y_test-np.mean(ypred_test_OLS,axis=1,keepdims=True))**2)
    Bias_Ridge_train[i]=np.mean((y_train-np.mean(ypred_train_Ridge,axis=1,keepdims=True))**2)
    Bias_Ridge_test[i]=np.mean((y_test-np.mean(ypred_test_Ridge,axis=1,keepdims=True))**2)
    Bias_Lasso_train[i]=np.mean((y_train-np.mean(ypred_train_Lasso,axis=1,keepdims=True))**2)
    Bias_Lasso_test[i]=np.mean((y_test-np.mean(ypred_test_Lasso,axis=1,keepdims=True))**2)

#%%

"""Plot sampled results"""

plt.figure()
plt.semilogx(lamda,MSE_boot_OLS_train,'b',label='Train Error')
plt.semilogx(lamda,MSE_boot_OLS_test,'--b',label='Test Error')
plt.semilogx(lamda,Var_OLS_train,'r',label='Train Variance')
plt.semilogx(lamda,Var_OLS_test,'--r',label='Test Variance')
plt.semilogx(lamda,Bias_OLS_train,'g',label='Train Bias')
plt.semilogx(lamda,Bias_OLS_test,'--g',label='Test Bias')
plt.legend()
plt.xlabel('Hyperparameters')
plt.ylabel('Errors')
plt.title('Bias-Variance Analysis (OLS)')


plt.figure()
plt.semilogx(lamda,MSE_boot_Ridge_train,'b',label='Train Error')
plt.semilogx(lamda,MSE_boot_Ridge_test,'--b',label='Test Error')
plt.semilogx(lamda,Var_Ridge_train,'r',label='Train Variance')
plt.semilogx(lamda,Var_Ridge_test,'--r',label='Test Variance')
plt.semilogx(lamda,Bias_Ridge_train,'g',label='Train Bias')
plt.semilogx(lamda,Bias_Ridge_test,'--g',label='Test Bias')
plt.legend()
plt.xlabel('Hyperparameters')
plt.ylabel('Errors')
plt.title('Bias-Variance Analysis (Ridge)')

plt.figure()
plt.semilogx(lamda,MSE_boot_Lasso_train,'b',label='Train Error')
plt.semilogx(lamda,MSE_boot_Lasso_test,'--b',label='Test Error')
plt.semilogx(lamda,Var_Lasso_train,'r',label='Train Variance')
plt.semilogx(lamda,Var_Lasso_test,'--r',label='Test Variance')
plt.semilogx(lamda,Bias_Lasso_train,'g',label='Train Bias')
plt.semilogx(lamda,Bias_Lasso_test,'--g',label='Test Bias')
plt.legend()
plt.xlabel('Hyperparameters')
plt.ylabel('Errors')
plt.title('Bias-Variance Analysis (Lasso)')




    
    
    
    

