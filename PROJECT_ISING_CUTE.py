# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:57:24 2020

@author: Patricia
"""

# %%  LIBRARIES USED IN THIS PROJECT 
import numpy as np
import scipy.sparse as sp
import sklearn.linear_model as skl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
import seaborn as sns
import scipy.linalg as scl
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn


# %% SETTINGS AND PARAMETERS

sns.set(color_codes=True)
cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
plt.close('all')

# Plotting parameter: =1 for plotting
plot_1=0

# Parameters for Ising Model : 
L=40                  #system size
Nstates=10000         #Number states
J=1                   #Coupling constant
n_bootstrap=20        #Number of bootstrap
n_pol = 5             # Order of the polynomial 

COMPLX=np.arange(n_pol)       #Complexity of the polynomial 
lamda=np.logspace(-3,-2,2)    #Define hyperparameters for Ridge and Lasso

# %% FUNCTIONS 

def ising_energies(spins,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',spins,J,spins)
    return E

def MSEdim (y_test,y_pred):
    e = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
    return e

def bias (y_test,y_pred):
    b = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
    return b

def var (y_test,y_pred):
    v = np.mean( np.var(y_pred, axis=1, keepdims=True) )
    return v


# %% DATA: TRAINING AND TEST
    
spins=np.random.choice([-1, 1], size=(Nstates,L))   # create 10000 random Ising states
energies=ising_energies(spins,L).reshape(-1,1)    # calculate Ising energies

x = spins
y = energies

# Split the data in train and test 
x_train, x_test, y_train, y_test  = train_test_split(x,y,test_size=0.2)


# DESIGN MATRIX

X_train = np.zeros((x_train.shape[0], L**2))  
for i in range(x_train.shape[0]):    
    X_train[i] = np.outer(x_train[i], x_train[i]).ravel()

X_test = np.zeros((x_test.shape[0], L**2))  
for i in range(x_test.shape[0]):   
    X_test[i] = np.outer(x_test[i], x_test[i]).ravel()

# First column filled with 1
    
X_train_own = np.concatenate(
    (np.ones(len(X_train))[:, np.newaxis], X_train),
    axis=1)

X_test_own = np.concatenate(
    (np.ones(len(X_test))[:, np.newaxis], X_test),
    axis=1)    

# %% LINEAR REGRESSION

"""
Linear Regression 
"""
 
clf_OLS = skl.LinearRegression().fit(X_train, y_train)  #MODEL
J_sk1 = np.array(clf_OLS.coef_).reshape((L, L))

# define error lists
train_errors_leastsq = []
test_errors_leastsq = []


y_train_OLS=clf_OLS.predict(X_train)    
y_test_OLS=clf_OLS.predict(X_test)
 
# Mean Squared Error and R2 for OLS
MSE_train=MSE(y_train,y_train_OLS) 
MSE_test=MSE(y_test,y_test_OLS)  
R2_train=R2(y_train,y_train_OLS)
R2_test=R2(y_test,y_test_OLS) 

print('Mean Sq Error Train Linear Regression')
print(MSE_train)
print('Mean Sq Error Test Linear Regression')
print(MSE_test)

print('R2 Train Lin Reg')
print(R2_train)
print('R2 test Lin Reg')
print(R2_test)

if plot_1:
    fig = plt.figure(figsize=(20, 14))
    im = plt.imshow(J_sk1, **cmap_args)
    plt.title("LinearRegression from Scikit-learn", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    cb = fig.colorbar(im)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=18)
    plt.show()


# %% REGRESSION WITH HYPERPARAMETER & BOOTSTRAP


#Initialize coeffficients for ridge regression and Lasso
coefs_leastsq = []
coefs_ridge = []
coefs_lasso=[]

# define error lists

train_errors_ridge = []
test_errors_ridge = []

train_errors_lasso = []
test_errors_lasso = []


# Initialization of the variables 
y_pred_Ridge=np.zeros((x_test.shape[0], n_bootstrap))
y_pred_Lasso=np.zeros((x_test.shape[0],n_bootstrap))

MSE_Ridge=np.zeros(len(lamda))
Bias_Ridge=np.zeros((len(lamda)))
Variance_Ridge=np.zeros((len(lamda)))

MSE_Lasso=np.zeros((len(lamda)))
Bias_Lasso=np.zeros((len(lamda)))
Variance_Lasso=np.zeros((len(lamda)))

# %%

"""


y_train_OLS=clf_OLS.predict(X_train)    
y_test_OLS=clf_OLS.predict(X_test)
 
# Mean Squared Error and R2 for OLS
MSE_train=MSE(y_train,y_train_OLS) 
MSE_test=MSE(y_test,y_test_OLS)  
R2_train=R2(y_train,y_train_OLS)
R2_test=R2(y_test,y_test_OLS) 
"""
# define error lists
train_errors_leastsq = []
test_errors_leastsq = []


for i in range(len(lamda)):
    
        ### ordinary least squares
    clf_OLS = skl.LinearRegression().fit(x_train, y_train)  #MODEL
#    J_sk1 = np.array(clf_OLS.coef_).reshape((L, L))

    # use the coefficient of determination R^2 as the performance of prediction.
    train_errors_leastsq.append(clf_OLS.score(x_train, y_train))
    test_errors_leastsq.append(clf_OLS.score(x_test,y_test))
    
    
# %%    
    for j in range(n_bootstrap):
        X_,y_=resample(X_train,y_train)
        
        clf_Ridge=skl.Ridge(alpha=lamda[i],fit_intercept=False).fit(X_,y_)
        y_pred_Ridge[:,j]=clf_Ridge.predict(X_test).flatten()
        J_sk2 = np.array(clf_Ridge.coef_).reshape((L, L))
        
        train_errors_ridge.append(clf_Ridge.score(x_train, y_train))
        test_errors_ridge.append(clf_Ridge.score(x_test,y_test))
        
        clf_Lasso=skl.Lasso(alpha=lamda[i],fit_intercept=False,max_iter=1E+4).fit(X_,y_)
        y_pred_Lasso[:,j]=clf_Lasso.predict(X_test).flatten()
        J_sk3 = np.array(clf_Lasso.coef_).reshape((L, L))
        
        train_errors_lasso.append(clf_Lasso.score(x_train, y_train))
        test_errors_lasso.append(clf_Lasso.score(x_test,y_test))
        
        
    MSE_Ridge[i] = MSEdim(y_test,y_pred_Ridge)
    MSE_Lasso[i] = MSEdim(y_test,y_pred_Lasso)
    
    Bias_Ridge[i] = bias(y_test, y_pred_Ridge)
    Bias_Lasso[i] = bias(y_test, y_pred_Lasso)
    
    Variance_Ridge[i]= var(y_test, y_pred_Ridge)
    Variance_Lasso[i]=var(y_test, y_pred_Lasso)
    

 
# %% COMPARATIVE PLOT: MSE, BIAS, VARIANCE
if plot_1:
    fig = plt.figure(figsize=(20, 14))
    im = plt.imshow(J_sk1, **cmap_args)
    plt.title("LinearRegression from Scikit-learn", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    cb = fig.colorbar(im)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=18)
    plt.show()
    


if plot_1:
    
    for i in range(len(lamda)):
        
        fig, axarr = plt.subplots(nrows=1, ncols=3)
        
        axarr[0].imshow(J_sk1,**cmap_args)
        axarr[0].set_title('OLS \n Train$=%.3f$, Test$=%.3f$'%(MSE_train[-1], MSE_test[-1]),fontsize=16)
        axarr[0].tick_params(labelsize=16)
        
        axarr[1].imshow(J_sk2,**cmap_args)
        axarr[1].set_title('Ridge $\lambda=%.4f$\n Train$=%.3f$, Test$=%.3f$' %(lamda,train_errors_ridge[-1],test_errors_ridge[-1]),fontsize=16)
        axarr[1].tick_params(labelsize=16)
        
        im=axarr[2].imshow(J_sk3,**cmap_args)
        axarr[2].set_title('LASSO $\lambda=%.4f$\n Train$=%.3f$, Test$=%.3f$' %(lamda,train_errors_lasso[-1],test_errors_lasso[-1]),fontsize=16)
        axarr[2].tick_params(labelsize=16)
        
        divider = make_axes_locatable(axarr[2])
        cax = divider.append_axes("right", size="5%", pad=0.05, add_to_figure=True)
        cbar=fig.colorbar(im, cax=cax)
        
        cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
        cbar.set_label('$J_{i,j}$',labelpad=15, y=0.5,fontsize=20,rotation=0)
        
        fig.subplots_adjust(right=2.0)
        
        plt.show()
        
   
#To quantify learning, we also plot the in-sample and out-of-sample errors
    
# Plot our performance on both the training and test data
plt.semilogx(lamda, train_errors_leastsq, 'b',label='Train (OLS)')
plt.semilogx(lamda, test_errors_leastsq,'--b',label='Test (OLS)')
plt.semilogx(lamda, train_errors_ridge,'r',label='Train (Ridge)',linewidth=1)
plt.semilogx(lamda, test_errors_ridge,'--r',label='Test (Ridge)',linewidth=1)
plt.semilogx(lamda, train_errors_lasso, 'g',label='Train (LASSO)')
plt.semilogx(lamda, test_errors_lasso, '--g',label='Test (LASSO)')

fig = plt.gcf()
fig.set_size_inches(10.0, 6.0)

#plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
#           linewidth=3, label='Optimum on test')
plt.legend(loc='lower left',fontsize=16)
plt.ylim([-0.1, 1.1])
plt.xlim([min(lamda), max(lamda)])
plt.xlabel(r'$\lambda$',fontsize=16)
plt.ylabel('Performance',fontsize=16)
plt.tick_params(labelsize=16)
plt.show()    