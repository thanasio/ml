import mnist_loader
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

# import input_data


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list (training_data)
n_iterations = 10
mid_layer=30
mini_batch_size = 10
n = len(training_data)
eta = 0.2

W_1 = np.random.randn(mid_layer, 784)
b_1 = np.random.randn(mid_layer, 1)
W_2 = np.random.randn(784, mid_layer)
b_2 = np.random.randn(784, 1)

MSE_arr = [0] * n_iterations

for iteration in range(n_iterations):
    mini_batches = [
        training_data[k:k + mini_batch_size]
        for k in range(0, n, mini_batch_size)]

    for mini_batch in mini_batches:

        for x, y in mini_batch:
            # y = x # Because we are implementing AE

            # FeedForward
            Z_0 = x
            A_0 = Z_0
            Z_1 = W_1.dot(A_0) + b_1
            A_1 = sigmoid(Z_1)
            Z_2 = W_2.dot(A_1) + b_2

            A_2 = sigmoid(Z_2) # Note we have not used sigmoid(Z_2), you know why? Because it is a linear regression and not logistic!

            error = (A_2 - x)
            #SE = np.square(error)
            # print(SE)
            #MSE = SE / len(A_2)
            #MMSE = np.sum(MSE)
            # print(MMSE)

            delta3 = (A_2 - x)
            delta_2 = 2 * delta3

            d_W_2 = delta_2.dot(A_1.T)  # derivative of 'J' w.r.t 'W2'
            d_b_2 = delta_2  # derivative of 'J' w.r.t 'b2'

            delta_1 = (W_2.T.dot(delta_2))*(sigmoid_prime(Z_1))

            d_W_1 = delta_1.dot(A_0.T)  # derivative of 'J' w.r.t 'W1'
            d_b_1 = delta_1  # derivative of 'J' w.r.t 'b1'

            W_1 = W_1 - eta * d_W_1
            W_2 = W_2 - eta * d_W_2

            b_1 = b_1 - eta * d_b_1
            b_2 = b_2 - eta * d_b_2

    # FeedForward - Testing model on the training dataset
    Z_0 = x
    # Z_0 =  np.reshape(Z_0, (784, 500))

    A_0 = Z_0
    Z_1 = W_1.dot(A_0) + b_1
    A_1 = sigmoid(Z_1)
    Z_2 = W_2.dot(A_1) + b_2
    A_2 = sigmoid(Z_2)


    err = A_2 - Z_0
    SE = np.square(err)
    print(iteration)
    # print(SE)
    MSE = SE / len(A_2)
    MMSE = np.sum(MSE)
    MSE_arr[iteration] = MMSE
    print(MMSE)

print('Finish')


