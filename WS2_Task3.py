import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from os import path

scaler = StandardScaler()
data = pd.read_csv(path.join("data", "housing.csv"), header=None)
data = data.to_numpy()
data = normalize(data, norm='l2')
X_b = data[0:13, ]
ones = np.ones((1, 506))
X_b1 = np.append(ones, X_b, axis=0)
X_b1 = X_b1.T
y = data[[13],:]
y = y.T

XTX_1 = np.linalg.inv(np.matmul(X_b1.T, X_b1))
XtY=np.matmul(X_b1.T, y)

m = np.matmul(XTX_1, XtY)  #1. Find m
y_predict = np.matmul( X_b1, m)       #2. Check model predictions
error = [abs(y[i] - y_predict[i]) for i in range(len(y))]            #3. Find error
print(error)
MSE= np.average(np.array([(y[i] - y_predict[i])*(y[i] - y_predict[i]) for i in range(len(y))] ))
print("MSE: "+str(MSE))

Network_weights = sorted(m.ravel())  #4. Print network weights, and explain
                    #   which variable has the most weight (significance) and why?
print(Network_weights)
print(np.argsort((m.ravel())))
print(m.ravel())