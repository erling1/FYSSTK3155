import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed()
n = 100

# Make data set. reshape the x into a coloumn vector with n rows
x = np.linspace(-3, 3, n).reshape(-1, 1)


y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

#Defining our design matrix to be a matrix of len(x)=n rows, and 5 coloumns

X = np.zeros((len(x),6))


#issue where x is a 1 cloumn matrix, whilst X[:1] is a flat array, fixed by using the command
#flatten() on the x-array so that it is the shape of a column-vector in X
X[:,0] = 1.0
X[:,1] = x.flatten()
X[:,2] = x.flatten()**2
X[:,3] = x.flatten()**3
X[:,4] = x.flatten()**4
X[:,5] = x.flatten()**5

# We split the data in test and training data. We specfify X
X_train, X_test, y_train, y_test = train_test_split(X, y)


# @ is the mulitpliction of matricies. 
#  Least squares fitting method involves finding beta. Without deriving the beta, the formula is:
# beta = (X^T * X)^-1 X^T y. One important note is that all X's and y must be of the same training or test set. 
# everything is matrices.

#This beta is the beta used for predicting the output of the trainingset. It is from the traingset we get beta. 

beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train


#predicted output values from the training set. This is y_tilde, which is the predicted outcome of our model 
y_predict_train = X_train @ beta
y_predict_test = X_test @ beta

#MSE between trained and predicted outcomes y
MSE_train = mean_squared_error(y_train,y_predict_train)

print("MSE_train", MSE_train)


#R2 between trained and predicted outcomes y
R2_train = r2_score(y_train,y_predict_train)

print("R2_train", R2_train)

#MSE between test and predicted outcomes y

MSE_test = mean_squared_error(y_test,y_predict_test)

print("MSE_test", MSE_test)


R2_test = r2_score(y_test, y_predict_test)

print("R2_test", R2_test )



#Setting now up a design matrix M which goes up to polynominal degree 15
#

#Y = np.zeros((len(x),16))
#Y[:,0] = 1.0
#Y[:,1] = x.flatten()

MSE = []
R2 = []

pdegree = np.zeros(16)

x_train, x_test, y_train, y_test = train_test_split(x,y) 


#looping over all the degrees, and for each one creating a feature, or a design matriz. 
#One way to do this, is to create a "pipeline" which first applies the design matrix, whilst afterwards applies the
# *estimator*, which calculates each beta, for each desing matrix
for  degree in range(16): 
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False)) #creating pipelin which applies PolyF, and LR
    model.fit(x_train,y_train) #fitting my model with my training data
    y_tr_predict = model.predict(x_train) #making predctions using  predict() and training input
    y_te_predict = model.predict(x_test) #making predctions using  predict() and test input
    pdegree[degree] = degree #creating a list where each element is a higher and higher degree
    MSE[degree] = mean_squared_error(y_train,y_tr_predict)
    print(MSE)
    #R2[degree] = r2_score(y,y_tr)
    #print(MSE)
    #printi(R2)



    







#Y[:,2] = x.flatten()**2
#Y[:,3] = x.flatten()**3
#Y[:,4] = x.flatten()**4
#Y[:,5] = x.flatten()**5
#Y[:,6] = x.flatten()**6
#Y[:,7] = x.flatten()**7
#Y[:,8] = x.flatten()**8
#Y[:,9] = x.flatten()**9
#Y[:,10] = x.flatten()**10
#Y[:,11] = x.flatten()**11
#Y[:,12] = x.flatten()**12
#Y[:,13] = x.flatten()**13
#Y[:,14] = x.flatten()**14
#Y[:,15] = x.flatten()**15


#beta_Y = np.linalg.inv(Y_train.T @ Y_train) @ Y_train.T @ y_train

#y_predict_train_Y = Y_train @ beta_Y
#y_predict_test_Y = Y_test @ beta_Y

#MSE_Y_train = mean_squared_error(y_train,y_predict_train_Y)
#MSE_Y_test = mean_squared_error(y_test, y_predict_test_Y)

#R2_test_Y = r2_score(y_test, y_predict_test_Y)
#R2_train_Y = r2_score(y_train,y_predict_train_Y)

#print("R2_test_Y", R2_test_Y)
#print("R2_train_Y", R2_train_Y)
#print(Y.shape)
#print(beta_Y.shape)

###
