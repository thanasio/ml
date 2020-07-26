import workshops.ws6.mnist_loader as mnist_loader
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

import workshops.ws6.input_data as input_data
import time

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


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data_list = list(training_data)
n_iterations = 10
mini_batch_size = 10
n = len(training_data_list)
eta = 0.1
middle_layer = 30
W_1 = np.random.randn(middle_layer, 784)
b_1 = np.random.randn(middle_layer, 1)
W_2 = np.random.randn(10, middle_layer)
b_2 = np.random.randn(10, 1)

t0 = time.time()

for iteration in range(n_iterations):
    mini_batches = [
        training_data_list[k:k + mini_batch_size]
        for k in range(0, n, mini_batch_size)]


    for mini_batch in mini_batches:

        for x, y in mini_batch:


            # ForwardProp

            Z_0 = x
            A_0 = Z_0
            Z_1 = W_1.dot(A_0)  + b_1
            A_1 = sigmoid(Z_1)
            Z_2 = W_2.dot(A_1) + b_2
            A_2 = sigmoid(Z_2)
            error = (A_2 - y)*(A_2-y)

            # Backprop

            delta3 = A_2 - y
            delta_2 = 2*delta3

            d_W_2 = delta_2.dot(A_1.T) # derivative of 'J' w.r.t 'W2'
            d_b_2 = delta_2  # derivative of 'J' w.r.t 'b2'

            delta_1 = (W_2.T.dot(delta_2) ) *(sigmoid_prime(Z_1))
            d_W_1 = delta_1.dot(A_0.T)  # derivative of 'J' w.r.t 'W1'
            d_b_1 = delta_1  # derivative of 'J' w.r.t 'b1'

            W_1 = W_1 - eta*d_W_1
            W_2 = W_2 - eta * d_W_2

            b_1 = b_1 - eta*d_b_1
            b_2 = b_2 - eta*d_b_2


    # Testing the trained model

t1 = time.time()
elapsed = t1-t0

print("NN took " + str(elapsed) + " seconds to run")

mnist = input_data.read_data_sets("MNIST_Data/", one_hot=False)
batch = mnist.train.next_batch(500)
tb = mnist.train.next_batch(100)

y_t = batch[1]
x_t = batch[0]

y_mat = oneHotIt(y_t)  # Next we convert the integer class coding into a one-hot representation
y_t = y_mat.T

#Z_0 = x_t[2,]
#Z_0 =  np.reshape(Z_0, (784, 1))
#Z_0 = xx = normalize(x, norm='l2')

# ForwardProp, testing on a different datasamples

A_0 = x_t
#A_0 =  np.reshape(Z_0, (784, 500))

Z_1 = W_1.dot(A_0.T)  + b_1
A_1 = sigmoid(Z_1)
Z_2 = W_2.dot(A_1) + b_2
A_2 = sigmoid(Z_2)

for i in range(A_2.shape[0]):
    for j in range(A_2.shape[1]):

        if A_2[i][j]<0.5:
            A_2[i][j]=0
        else:
            A_2[i][j]=1


accuracy = sum(A_2 == y_t) / (float(len(y_t)))

acc = sum(accuracy)/len(accuracy)
print(acc)
print('Finish')