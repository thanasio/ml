import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import random
import workshops.ws5.input_data
from workshops.ws5 import mnist_loader, input_data


def h(Theta, X):
    thetax = np.dot(X, Theta)
    return sigmoid(thetax)

def oneHotIt(Y):
    m = Y.shape[0]
    #Y = Y[:,0]
    OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    OHX = np.array(OHX.todense()).T
    return OHX

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


# mnist = input_data.read_data_sets("MNIST_Data/", one_hot=False)
# batch = mnist.train.next_batch(5000)
# tb = mnist.train.next_batch(1000)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#

X_test = []
Y_test = []
for x, y in test_data:
    X_test.append( x.ravel())
    Y_test.append( y)

X_test = np.array(X_test)
Y_test = oneHotIt(np.array(Y_test))

training_data_list = list(training_data)

n_iterations = 300
mini_batch_size = 50
m = len(training_data_list)

theta = np.random.randn(10,784)
eta=0.1

for iteration in range(n_iterations):
    mini_batches = [ training_data_list[k:k + mini_batch_size] for k in range(0, m, mini_batch_size)]
    # counter=0
    for mini_batch in mini_batches:
        X_b1 = []
        Y = []
        # idx = random.randint(mini_batch_size)
        for x, y in mini_batch:
            X_b1.append(x.ravel())
            Y.append(y.ravel())
        X_b1 = np.array(X_b1)
        Y = np.array(Y)

        for i in range(10):
            # calculate h of theta h(theta)
            h_theta_x = h(theta[i], X_b1)
            # calculate step error
            err = h_theta_x - Y.T[i]
            # calculate new gradients
            gradients = (2 / m) * np.dot(X_b1.T, err)
            # calculate new theta
            theta[i] = theta[i] - eta * gradients

            # print('minibatch: ' + str(counter))
            # counter += 1


    # Testing the trained model on test_data

y_predict = np.array([h(theta[i], np.array(X_test)) for i in range(10)])

for i in range(y_predict.shape[0]):
    for j in range(y_predict.shape[1]):

        if y_predict[i][j]<0.5:
            y_predict[i][j]=0
        else:
            y_predict[i][j]=1


results = y_predict.T == Y_test
accuracies = [sum(x) for x in results]
accuracy = sum([x==10 for x in accuracies]) / (float(len(Y_test)))
print(accuracy)

print('Finish')


