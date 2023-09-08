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


#we rescale our feature matrix by subtracting the mean value of each coloumn to each coloumn-element

New_X = np.zeros((len(x),6))
X_10 = np.zeros((len(x),11))
X_10[:,0] = 1.0
X_15 = np.zeros((len(x),16))
X_15[:,0] = 1.0

New_X[:,0]  = 1
for i in range(6):
    New_X[:,i] = X[:,i] - np.mean(X[:,i]) #er ikke forskjell på matrisene ettersom mean er så liten?


for i in range(11):
    if i != 0:
        
        X_10[:,i] = x.flatten()**i

for i in range(16):
    if i != 0:
        
        X_15[:,i] = x.flatten()**i

# We split the data in test and training data. We specfify X
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_new_train, X_new_test, y_train, y_test = train_test_split(New_X, y)

X_10_train, X_10_test, y_train, y_test = train_test_split(X_10, y)

X_15_train, X_15_test, y_train, y_test = train_test_split(X_15, y)


# @ is the mulitpliction of matricies. 
#  Least squares fitting method involves finding beta. Without deriving the beta, the formula is:
# beta = (X^T * X)^-1 X^T y. One important note is that all X's and y must be of the same training or test set. 
# everything is matrices.

#This beta is the beta used for predicting the output of the trainingset. It is from the traingset we get beta. 

beta_ols = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train


#predicted output values from the training set. This is y_tilde, which is the predicted outcome of our model 
y_predict_train = X_train @ beta_ols
y_predict_test = X_test @ beta_ols

#MSE between trained and predicted outcomes y
MSE_train = mean_squared_error(y_train,y_predict_train)

print("MSE_train_ols", MSE_train)


#R2 between trained and predicted outcomes y
R2_train = r2_score(y_train,y_predict_train)

#print("R2_train", R2_train)

#MSE between test and predicted outcomes y

MSE_test = mean_squared_error(y_test,y_predict_test)

print("MSE_test_ols", MSE_test)


R2_test = r2_score(y_test, y_predict_test)

#print("R2_test", R2_test )


lambdaa = [0.0001, 0.001, 0.01, 0.1, 1 ]
Identity = np.eye(6)
Identity2 = np.eye(11)
Identity3 = np.eye(16)
MSE_test_ridge = []
MSE_train_ridge = []
difference = []

MSE_ridge_10_test = []
MSE_ridge_10_train = []
MSE_ridge_15_test = []
MSE_ridge_15_train = []

for i in lambdaa:
    beta_ridge = np.linalg.inv(X_new_train.T @ X_new_train + i * Identity) @ X_new_train.T @ y_train
    beta_ridge_10 = np.linalg.inv(X_10_train.T @ X_10_train + i * Identity2) @ X_10_train.T @ y_train
    beta_ridge_15 = np.linalg.inv(X_15_train.T @ X_15_train + i * Identity3) @ X_15_train.T @ y_train

    y_predicted_test_ridge = X_new_test @ beta_ridge
    y_predicted_train_ridge = X_new_train @ beta_ridge

    y_predicted10_test_ridge = X_10_test @ beta_ridge_10
    y_predicted10_train_ridge = X_10_train @ beta_ridge_10

    y_predicted15_test_ridge = X_15_test @ beta_ridge_15
    y_predicted15_train_ridge = X_15_train @ beta_ridge_15

    MSE_test_ridge.append(mean_squared_error(y_test,y_predicted_test_ridge))
    MSE_train_ridge.append(mean_squared_error(y_train,y_predicted_train_ridge))

    MSE_ridge_10_test.append(mean_squared_error(y_test,y_predicted10_test_ridge))
    MSE_ridge_10_train.append(mean_squared_error(y_train,y_predicted10_train_ridge))

    MSE_ridge_15_test.append(mean_squared_error(y_test,y_predicted15_test_ridge))
    MSE_ridge_15_train.append(mean_squared_error(y_train,y_predicted15_train_ridge))

    
    
#Looks like OLS is better at degree 5, but Ridge becomes just as good at deg=10, and better at deg=15. lambda does not change notably.


fig, (ax1,ax2) = plt.subplots(1,2)

ax1.plot(lambdaa,MSE_ridge_15_train, label = 'MSE_DEG15', color = 'red')
ax1.set_title('MSE vs Lambda')

ax2.plot(lambdaa,MSE_ridge_10_train, label = 'MSE_DEG10', color = 'green')
ax2.set_title('MSE vs Lambda')

ax1.legend()
ax2.legend()

table_of_lambda_MSE_10_test = [[lambdaa[i],MSE_ridge_10_test[i]] for i in range(len(lambdaa))]

for i in table_of_lambda_MSE_10_test:
    formatted_row = "Lambda: {:.4f}, MSE_10_ridge: {:.5f}".format(i[0], i[1])
    print(formatted_row)
    
table_of_lambda_MSE_15_test = [[lambdaa[i],MSE_ridge_15_test[i]] for i in range(len(lambdaa))]

for i in table_of_lambda_MSE_15_test:
    formattedd_row = "Lambda: {:.4f}, MSE_15_ridge: {:.5f}".format(i[0], i[1])
    print(formattedd_row)





plt.tight_layout()#we see that the MSE is linearly dependent on lambda, and for increasing lambda, leads to worse and worse fits.
plt.show()
