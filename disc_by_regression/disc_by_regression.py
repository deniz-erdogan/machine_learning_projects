import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safelog(x):
  return np.log(x + 1e-100)

def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))


images = np.loadtxt("hw03_data_set_images.csv",delimiter = ",")
y = np.loadtxt('hw03_data_set_labels.csv', delimiter = ",").astype(int)
print(np.shape(images))

K = max(y)

training_images = np.stack(([images[i*39 : i*39+25] for i in range(int(K))]), axis=0)
print(np.shape(training_images))

test_images = np.stack(([images[i*39+25 : i*39+39] for i in range(int(K))]), axis=0)
print(np.shape(test_images))

training_labels = np.stack(([y[i*39 : i*39+25] for i in range(int(K))]), axis=0)
print(np.shape(training_labels))

test_labels = np.stack(([y[i*39+25 : i*39+39] for i in range(int(K))]), axis=0)
print(np.shape(test_labels))

print(np.size(test_labels))


# Parameter Initialization
eta = 1e-03
epsilon = 1e-03
print(np.shape(training_images)[2])
w = np.random.uniform(low = -0.01, high = 0.01, size = (training_images.shape[2], 5))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1,5))

print(w)
print(w0)

# Gradients
def gradient_w(X, y_truth, y_hat):
    return(np.asarray([-np.matmul(y_truth[:,c] - y_hat[:,c], X) for c in range(K)]).transpose())

def gradient_w0(y_truth, y_hat):
    return(-np.sum(y_truth - y_hat, axis=0))


def calculate_difference(w, w_old, w0, w0_old):
    return np.sqrt(np.sum((w0 - w0_old) ** 2) + np.sum((w - w_old) ** 2))


y_truth = np.zeros((125, 5))
training_labels = training_labels.reshape(125,1)
for i in range(np.shape(y_truth)[0]):
    y_truth[i][training_labels[i]-1] = 1



training_images = training_images.reshape(125, 320)
iterations = 1
errors = []

while True:
    y_hat = sigmoid(training_images, w, w0)
    errors = np.append(errors, np.sum((0.5)*((y_truth - y_hat)**2)))

    w_old = w
    w0_old = w0


    w = w - eta * gradient_w(training_images, y_truth, y_hat)
    w0 = w0 - eta * gradient_w0(y_truth, y_hat)
    

    if calculate_difference(w, w_old, w0, w0_old) < epsilon:
        print(np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((w - w_old)**2)))
        print("Model converged.")
        break
    
    iterations += 1
        
print(iterations, "iterations were made.")


plt.figure(figsize = (10, 6))
plt.plot(range(len(errors)), errors, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# calculate confusion matrix
y_hat = np.argmax(y_hat, axis=1) + 1
y_truth = np.argmax(y_truth, axis=1) + 1
confusion_matrix = pd.crosstab(y_hat, y_truth, rownames = ["y_pred"], colnames = ["y_truth"])
print("Training Set Confusion Matrix:")
print(confusion_matrix)


test_set_assignments = sigmoid(test_images.reshape(5*14, 320), w, w0)
test_truth_labels = np.zeros((14*5, 5))
test_labels = test_labels.reshape(14*5)
for i in range(np.shape(test_labels)[0]):
    test_truth_labels[i][test_labels[i]-1] = 1

test_set_assignments = np.argmax(test_set_assignments, axis = 1) + 1
test_truth_labels = np.argmax(test_truth_labels, axis = 1) + 1

confusion_matrix = pd.crosstab(test_set_assignments, test_truth_labels, rownames = ['y_pred'], colnames = ['y_truth'])
print("Test Set confusion matrix:")
print(confusion_matrix)

