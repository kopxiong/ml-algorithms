# -*- coding: utf-8 -*-

# https://github.com/sourabhdattawad/Linear-regression-from-scratch-in-python

import numpy as np
import random
import sklearn
import pylab
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets.samples_generator import make_regression


class linearRegression:
    def cost(self, X, y, t0, t1):
        m = len(X)
        J = 1/(2*m) * sum([(t0 + t1 * X[i] - y[i]) ** 2 for i in range(m)])

        return J

    def gradientDescent(self, alpha, X, y, t0, t1, eps=0.001, max_iter=1000):
        converged = False
        iter = 0
        m = len(X)                          # Number of samples
        t0 = np.random.random(X.shape[1])   # Initial value of theta0
        t1 = np.random.random(X.shape[1])   # Initial value of theta1

        J = 1/(2*m) * sum([(t0 + t1 * X[i] - y[i]) ** 2 for i in range(m)])     #Initial Error

        while not converged:
            grad0 = 1.0/m * (sum([(t0 + t1 * X[i] - y[i]) for i in range(m)]))
            grad1 = 1.0/m * (sum([(t0 + t1 * X[i] - y[i]) * X[i] for i in range(m)]))

            t0 = t0 - alpha * grad0
            t1 = t1 - alpha * grad1

            error = self.cost(X, y, t0, t1)

            # If error difference of current and prev is less than some threshold: here(0.0001)
            if abs(J - error) < 0.0001:
                print("Converged successfully")
                converged = True

            J = error

        return t0, t1

if __name__ == '__main__':
    # Dummy dataset (tried to use dataset in Kaggle, but suffered from
    # RuntimeWarning: overflow encountered in square)
    X, y = make_regression(n_samples=100, n_features=1, n_informative=1, random_state=0, noise=35)

    # some initial values
    alpha  = 0.01
    eps    = 0.01
    t0, t1 = 0, 0

    print("Initial scatter plot")
    plt.scatter(X, y)
    plt.show()

    lr_classifier = linearRegression()
    theta0, theta1 = lr_classifier.gradientDescent(alpha, X, y, t0, t1, eps, max_iter=1000)
    print('theta0 = %s theta1 = %s' %(theta0, theta1))

    print("Line plot after linear regression")
    plt.scatter(X, y)
    plt.plot(X, theta0 + X * theta1, 'r')
    plt.show()
