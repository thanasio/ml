import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import scipy.sparse
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())
X = iris["data"] # petal width
X = normalize(X, norm='l2')
y = iris["target"]
m = y.shape[0]

def h(Theta, X):
    thetax = np.matmul(Theta.T, X)
    return 1/(1+np.exp(-thetax))

def oneHotIt(Y):
    m = Y.shape[0]
    #Y = Y[:,0]
    OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    OHX = np.array(OHX.todense()).T
    return OHX


y_mat = oneHotIt(y)  # Next we convert the integer class coding into a one-hot representation
num_classes=y_mat.shape[1]
y=y_mat
X_b = X.T
ones = np.ones((1, 150))
X_b1 = np.append(ones, X_b, axis=0)
X_b1_t = X_b1.T
n_iterations = 50000

eta =0.01 # learning rate
m =X.shape[0]
theta = np.random.rand(3,5)

for iteration in range(n_iterations):
    for i in range(num_classes):
        # calculate h of theta h(theta)
        h_theta_x = h(theta[i], X_b1)
        # calculate step error
        err = h_theta_x - y.T[i]
        # calculate newgreadients
        gradients = (2 / m) * np.matmul(X_b1, err.T)
        # calculate new theta
        theta[i] = theta[i] - eta * gradients

theta_best =theta #fil it in
y_predict = np.array([h(theta[i], X_b1) for i in range(num_classes)]) #fil it in

y_plot = y.T
for i in range(num_classes):
    plt.plot(y_predict[i], label="predicted")

    plt.plot(y_plot[i], label="actual")
    plt.xlabel('x - axis')
    # Set the y axis label of the current axis.
    plt.ylabel('y - axis')
    # Set a title of the current axes.
    plt.title('predicted vs actual: class ' + str(i))
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.savefig('task_4_2_'+str(i)+'.png')
    plt.close()

for i in range(y_predict.shape[0]):
    for j in range(y_predict.shape[1]):

        if y_predict[i][j]<0.5:
            y_predict[i][j]=0
        else:
            y_predict[i][j]=1


error = sum([1 if y_predict[j][i] != y.T[j][i] else 0 for i in range (y_predict.shape[1]) for j in range(y_predict.shape[0])])

# calculate acccuracy
num_samples = y_predict.shape[0]*y_predict.shape[1]
accuracy =((num_samples - error) / num_samples)

print(accuracy)
print(error)






