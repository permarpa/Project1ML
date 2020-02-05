# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:24:47 2020

@author: acer pc
"""

"""Project 1 - Ising Model - Part (a)+(b)"""

"""Import all necessary Python directories"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.model_selection import train_test_split as splitter
from sklearn.utils import resample
import seaborn as sns

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
    return X

#Define function to calculate the learned interaction
    
def Learned_Interaction(coeff):
    IntMat=np.reshape(coeff,(40,40))
    return IntMat
     
#Construct the design matrix
                    
X=Design_Matrix(state)

#Split the data into training and test

X_train,X_test,y_train,y_test=splitter(X,Energy,test_size=0.2)

#Define vector of hyperparameter values for OLS< Ridge and Lasso and perfirm regression

lamda=np.logspace(-8,-1,8)

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
    

"""Plot data"""

plt.figure()
plt.semilogx(lamda,MSE_OLS_train,label='Train (OLS)',color='blue')
plt.semilogx(lamda,MSE_OLS_test,label='Test (OLS)',color='blue',marker='+')
plt.semilogx(lamda,MSE_Ridge_train,label='Train (Ridge)',color='red')
plt.semilogx(lamda,MSE_Ridge_test,label='Test (Ridge)',color='red',marker='+')
plt.semilogx(lamda,MSE_Lasso_train,label='Train (Lasso)',color='green')
plt.semilogx(lamda,MSE_Lasso_test,label='Test (Lasso)',color='green',marker='+')
plt.legend()
plt.xlabel('Hyperparameters')
plt.ylabel('MSE')
plt.title('MSE')
plt.show()

plt.figure()
plt.semilogx(lamda,R2_OLS_train,label='Train (OLS)',color='blue')
plt.semilogx(lamda,R2_OLS_test,label='Test (OLS)',color='blue',marker='+')
plt.semilogx(lamda,R2_Ridge_train,label='Train (Ridge)',color='red')
plt.semilogx(lamda,R2_Ridge_test,label='Test (Ridge)',color='red',marker='+')
plt.semilogx(lamda,R2_Lasso_train,label='Train (Lasso)',color='green')
plt.semilogx(lamda,R2_Lasso_test,label='Test (Lasso)',color='green',marker='+')
plt.legend()
plt.xlabel('Hyperparameters')
plt.ylabel('R2')
plt.title('R2')
plt.show()

plt.figure()
ax=sns.heatmap(OLS_IntMat)
plt.title('Lamda=1e-4, OLS')
plt.show()

plt.figure()
ax=sns.heatmap(Ridge_IntMat)
plt.title('Lamda=1e-4, Ridge')
plt.show()
                    
plt.figure()
ax=sns.heatmap(Lasso_IntMat)
plt.title('Lamda=1e-4, Lasso')
plt.show()


