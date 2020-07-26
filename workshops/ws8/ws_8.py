X=[10, 40, 18, 54]
Y=[41, 15, 29, 24]


def covariance(x, y):
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    cov = 0
    for i in range(len(x)):
        cov += (x[i] - x_mean) * (y[i] - y_mean) / (n - 1)

    return cov


cov = covariance(X, Y)

print(cov)

X=[1, -1, 4]
Y=[2, 1, 3]
Z=[1, 3, -1]

cov_xy = covariance(X, Y)
cov_xz = covariance(X, Z)
cov_yz = covariance(Y, Z)

print(cov_xy)
print(cov_xz)
print(cov_yz)

# negative covariance, means that when the x set increases, the other decreases and vice versa.

import numpy as np
from numpy import linalg as LA


def PCA(x, ndims):
    covariance_mat = np.cov(x)
    eigenvalues, eigenvectors = LA.eig(covariance_mat)
    indexes = np.argsort(eigenvalues)
    rows=[]
    for idx in indexes[:ndims]:
        v = eigenvectors[idx]
        row = np.matmul(v.T,x)
        rows.append(row)
    return rows


data = [X, Y, Z]

pca = PCA(data, 2)
print(pca)