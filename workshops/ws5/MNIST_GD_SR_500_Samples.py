import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from sklearn.preprocessing import normalize

from workshops.ws5 import input_data

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def h(Theta, X):
    thetax = np.dot(X, Theta)
    return sigmoid(thetax)

mnist = input_data.read_data_sets("../../MNIST_Data/", one_hot=False)
batch = mnist.train.next_batch(500)
tb = mnist.train.next_batch(100)

y = batch[1]
x = batch[0]
X_b1 = np.c_[ x, np.ones(500) ]

y_t = tb[1]
x_t = tb [0]
x_t = np.c_[ x_t, np.ones(100) ]
#x = normalize(x, norm='l2')

def oneHotIt(Y):
    m = Y.shape[0]
    #Y = Y[:,0]
    OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    OHX = np.array(OHX.todense()).T
    return OHX

y_mat = oneHotIt(y)
y=y_mat


y_t = oneHotIt(y_t)

#X_b1 = X_b1.T


n_iterations = 300

eta = 0.1 # learning rate
m = y.shape[0]

theta = np.random.randn(10,785)
print('run the alg')
for iteration in range(n_iterations):

    for i in range(10):
        # calculate h of theta h(theta)
        theta_x = np.dot(X_b1, theta[i])
        h_theta_x = sigmoid(theta_x)
        # calculate step error
        err = h_theta_x - y.T[i]
        # calculate new gradients
        gradients = (2 / m) * np.dot(X_b1.T, err)
        # calculate new theta
        theta[i] = theta[i] - eta * gradients
    #y_predict = X_b1.dot(theta)


y_predict = np.array([h(theta[i], np.array(x_t)) for i in range(10)])



y_plot = y.T
for i in range(10):

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
    plt.savefig('task_5_1_'+str(i)+'.png')
    plt.close()

for i in range(y_predict.shape[0]):
    for j in range(y_predict.shape[1]):

        if y_predict[i][j]<0.5:
            y_predict[i][j]=0
        else:
            y_predict[i][j]=1


results = y_predict.T == y_t
accuracies = [sum(x) for x in results]
accuracy = sum([x==10 for x in accuracies]) / (float(len(y_t)))
print(accuracy)

print('Finish')
# do this for train - test
# check previous files for xtheta and transpositions
# add bias
# check results (both)