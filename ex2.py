import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt



#Exercise 2: making your own data and exploring scikit-learn


#generating random x-values, and a second-order polynominal with noise. 
x = np.random.rand(100,1)
y = 2.0 + 5*x*x + 10*np.random.randn(100,1)

#defining a linear regression model which fits the data and retro-actively determins the free parameters of the polynominal.
#I.e. LinearReg does not know the form of y, but does reproduce the same result due to it identifying it to be a second order 
#poly from the data-points
reg = LinearRegression()
reg.fit(x,y)


#Creating a 2D array, I.E x = 0, x = 1, y(x=0), y(x=1) 
x_line = np.array([[0],[1]])

#The linearregression model predicting y-values based on the datepoints.
#MSE need the y_values, and predicted y-values to be of the same sized array, so used same x-values.
y_prediction = reg.predict(x)

#calculating the difference between the actual y-values, and the predicted y_values

MSE = mean_squared_error(y,y_prediction)
r2 = r2_score(y,y_prediction)

print("MSE:", MSE)
print("R2:", r2)


#plt.plot(x,y,"ro")
#plt.plot(x_line,prediction, "r-")
#plt.show()

#print(y)
#print(y_prediction)