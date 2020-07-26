import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
scaler = StandardScaler()
from numpy import genfromtxt

data = pd.read_csv("data/housing.csv", header=None)
data = data.to_numpy()
data = normalize(data, norm='l2')
X_b = data[0:13, ]
ones = np.ones((1, 506))
X_b1_t = np.append(ones, X_b, axis=0)
X_b1 = X_b1_t.T
y = data[[13],:]
y = y.T

n_iterations = 100000
eta =  0.01# learning rate
m = data.shape[1]  # No of examples
theta = np.random.randn(14,1)
best_MSE = 1000000000
for iteration in range(n_iterations):
    X_b1_theta = np.matmul(X_b1 , theta)
    gradients = (2/m) * np.matmul(X_b1_t , ( X_b1_theta - y))
    theta = theta - eta*gradients
    #theta_best =
    y_predict = np.matmul( X_b1, theta)
    MSE= np.average(np.array([(y[i] - y_predict[i])*(y[i] - y_predict[i]) for i in range(len(y))] ))

    if MSE < best_MSE:
        best_MSE = MSE
        theta_best = theta
    if iteration % 10000 == 0:
        print(str(iteration) + ": " + str(MSE))
        print(theta_best.ravel())
print(theta_best)

