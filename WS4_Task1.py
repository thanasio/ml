import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

from sklearn import datasets


# pre-processing: L2 normalization normalizes values between 0.0-1.0
# creation of b vector, adds bias to the theta_x

def h(Theta, X):
    thetax = np.matmul(Theta.T, X)
    return 1/(1+np.exp(-thetax))
iris = datasets.load_iris()
list(iris.keys())
X = iris["data"] # petal width
X=X[0:99]
X = normalize(X, norm='l2')

y = iris["target"]
y=np.reshape(y, (1, 150)) #
y=y.T
y=y[0:99]
y = normalize(y, norm='l2')
X_b = X.T
ones = np.ones((1, 99))
X_b1 = np.append(ones, X_b, axis=0)
X_b1_t = X_b1.T


n_iterations = 50000
eta =0.01 # learning rate
m =X.shape[0]
theta = np.random.rand(5,1)

for iteration in range(n_iterations):
    # calculate h of theta h(theta)
    h_theta_x = h(theta, X_b1)
    # calculate step error
    err=h_theta_x - y.T
    # calculate newgreadients
    gradients = (2 / m) * np.matmul(X_b1, err.T)
    # calculate new theta
    theta = theta - eta*gradients

# choose best theta
theta_best =theta

# calculate predictions
y_predict =h(theta, X_b1)

y_pred = y_predict.ravel()
y2 = y.ravel()

plt.plot(y_pred, label = "predicted")

plt.plot(y2, label = "actual")
plt.xlabel('x - axis')
# Set the y axis label of the current axis.
plt.ylabel('y - axis')
# Set a title of the current axes.
plt.title('predicted vs actual')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.savefig('task_4_1.png')
#generate predictions out of probabilities
for i in range(y_predict.shape[0]):
    for j in range(y_predict.shape[1]):

        if y_predict[i][j]<0.5:
            y_predict[i][j]=0
        else:
            y_predict[i][j]=1

# sum of errored predictions
error = sum([1 if y_predict[0][i] != y[0][i] else 0 for i in range (len(y[0]))])

# calculate acccuracy
accuracy =((len(y) - error) / len(y))

print(accuracy)

