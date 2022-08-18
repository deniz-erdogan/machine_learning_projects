"""
Author: Deniz Erdogan
ID: 0069572
"""

import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt

# read data into memory
images = np.genfromtxt("hw06_data_set_images.csv", delimiter = ",")
labels = np.genfromtxt("hw06_data_set_labels.csv", delimiter = ",")


# get X and y values
X_train = images[:1000, :]
X_test = images[1000:, :]

y_train = labels[:1000].astype(int)
y_test = labels[1000:].astype(int)


# get number of samples and number of features
N_train = len(X_train)
D_train = X_train.shape[1]

print(X_train[0, :500])

# define Gaussian kernel function
def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D**2 / (2 * s**2))
    return(K)

print(X_train.shape)

number_of_bins = 64
histogram_train = np.zeros((np.shape(X_train)[0], number_of_bins))
histogram_test = np.zeros((np.shape(X_test)[0], number_of_bins))

for i in range(np.shape(X_train)[0]):
  for j in range(np.shape(X_train)[1]):
    if X_train[i][j] == 0:
      histogram_train[i][0] += 1
    else:
      histogram_train[i, int((X_train[i][j]+0.1) // 4)] += 1

    if X_test[i][j] == 0:
      histogram_test[i][0] += 1
    else:
      histogram_test[i, int((X_test[i][j]+0.1) // 4)] += 1


histogram_train = histogram_train / np.shape(X_train)[1]
histogram_test = histogram_test / np.shape(X_test)[1]

print("Histogram samples for the data:")
print(histogram_train[0:5, 0:5])
print(histogram_test[0:5, 0:5])

# PART 4
def intersection_kernel(h1, h2):
    dimension1 = h1.shape[0]
    dimension2 = h2.shape[0]
    matrix = np.zeros((dimension1,dimension2));

    for i in range(dimension1):
        for j in range(dimension2):
            matrix[i][j] = np.sum(np.min([h1[i], h2[j]],axis = 0))

    return matrix


K_train = intersection_kernel(histogram_train, histogram_train)
K_test = intersection_kernel(histogram_test, histogram_train)
print(K_train[0:5, 0:5])
print(K_test[0:5, 0:5])

# PART 5
yyK = np.matmul(y_train[:,None], y_train[None,:]) * K_train
C = 10
epsilon = 0.001

P = cvx.matrix(yyK)
q = cvx.matrix(-np.ones((N_train, 1)))
G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
A = cvx.matrix(1.0 * y_train[None,:])
b = cvx.matrix(0.0)   

# use cvxopt library to solve QP problems
result = cvx.solvers.qp(P, q, G, h, A, b)
alpha = np.reshape(result["x"], N_train)
alpha[alpha < C * epsilon] = 0
alpha[alpha > C * (1 - epsilon)] = C

# find bias parameter
support_indices, = np.where(alpha != 0)
active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
w0 = np.mean(y_train[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))

# calculate predictions on training samples
f_predicted = np.matmul(K_train, y_train[:,None] * alpha[:,None]) + w0
y_predicted = 2 * (f_predicted > 0.0) - 1
confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_train), y_train,
                               rownames = ["y_predicted"], colnames = ["y_train"])
print(confusion_matrix)

# calculate predictions on test samples
f_predicted = np.matmul(K_test, y_train[:,None] * alpha[:,None]) + w0
y_predicted = 2 * (f_predicted > 0.0) - 1
confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_train), y_test,
                               rownames = ["y_predicted"], colnames = ["y_test"])
print(confusion_matrix)

# PART 6
powers = np.array(range(-2,7))/2
training_accuracy = np.zeros(powers.shape)
test_accuracy = np.zeros(powers.shape)

for i,power in enumerate(powers):
    C = 10**power
    epsilon = 0.001

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_train, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
    h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
    A = cvx.matrix(1.0 * y_train[None,:])
    b = cvx.matrix(0.0)
    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_train)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C
    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y_train[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))
    
    train_predictions = np.matmul(K_train, y_train[:,None] * alpha[:,None]) + w0
    y_train_predicted = 2 * (train_predictions > 0.0) - 1
    training_confusion_matrix = pd.crosstab(np.reshape(y_train_predicted, N_train), y_train)
    training_accuracy[i] = (training_confusion_matrix[-1][-1]+training_confusion_matrix[1][1])/N_train
    
    test_predictions = np.matmul(K_test, y_train[:,None] * alpha[:,None]) + w0
    y_test_predicted = 2 * (test_predictions > 0.0) - 1
    test_confusion_matrix = pd.crosstab(np.reshape(y_test_predicted, N_train), y_test)
    test_accuracy[i] = (test_confusion_matrix[-1][-1]+test_confusion_matrix[1][1])/N_train
    
    
fig, axs = plt.subplots(1, figsize=(10,6))
axs.plot(10**powers,training_accuracy,"b-o", label = "training")
axs.plot(10**powers,test_accuracy,"r-o", label = "test")
axs.set_xlabel("Regularization Parameter (C)")
axs.set_ylabel("Accuracy")
axs.legend()
axs.set_xscale('log')

plt.show()