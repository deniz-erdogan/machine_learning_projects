# Deniz Erdogan 0069572
# ENGR421 HW4
# Nonparametric Regression

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


# %%
training_set = np.genfromtxt("hw04_data_set_train.csv", delimiter = ",")
test_set = np.genfromtxt("hw04_data_set_test.csv", delimiter = ",")

# get x and y values
x_train = training_set[:,0]
y_train = training_set[:,1]

x_test = test_set[:,0]
y_test = test_set[:,1]


# get number of classes and number of samples
K = np.max(y_train)
N = training_set.shape[0]

# %%
# Regressogram

h = 0.1 
origin = 0.0

min_x = min(x_train)
max_x = max(x_train)
data_interval = np.linspace(origin, max_x, int((max_x-origin)*1000))
left_borders = np.arange(origin, max_x, h)
right_borders = np.arange(origin+h, max_x+h, h)
p_hat = np.zeros(len(left_borders))

for b in range(len(left_borders)):
    p_hat[b]= np.sum(((left_borders[b] < x_train) & (x_train <= right_borders[b]))*y_train) / np.sum((left_borders[b] < x_train) & 
    (x_train <= right_borders[b]))


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))

ax1.plot(x_train, y_train, "b.", markersize = 10, label = "Training")
ax2.plot(x_test, y_test, "r.", markersize = 10, label = "Test")

ax1.set_xlabel("Time (sec)")
ax2.set_xlabel("Time (sec)")
ax1.legend(loc="upper right")
ax2.legend(loc="upper right")

ax1.set_ylabel("Signal (millivolt)")
ax2.set_ylabel("Signal (millivolt)")

for b in range(len(left_borders)):
    ax1.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
    ax2.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")

for b in range(len(left_borders) - 1):
    ax1.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")   
    ax2.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")   


plt.show()


# Calculate RMSE

rmse = 0
for i in range(len(left_borders)):
    for j in range(len(y_test)):
        if left_borders[i] < x_test[j] and x_test[j] <= right_borders[i]:
            rmse += ((y_test[j] - p_hat[i])**2)
rmse = np.sqrt(rmse/np.shape(y_test)[0])

print("Regressogram => RMSE is", rmse, "when h is", h)

# %%
# Running Mean Smoother
h = 0.1

def my_w(x, xi, h):
    return (1 * (np.abs(x-xi)/h <= 0.5))

rms = np.zeros(1001)
data_interval = np.linspace(0, 2, 1001)

for i in range(len(data_interval)):
    rms[i] = np.sum(my_w(data_interval[i], x_train, h) * y_train) / np.sum(my_w(data_interval[i], x_train, h))


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))

ax1.plot(x_train, y_train, "b.", markersize = 10, label = "Training")
ax2.plot(x_test, y_test, "r.", markersize = 10, label = "Test")

ax1.set_xlabel("Time (sec)")
ax2.set_xlabel("Time (sec)")
ax1.legend(loc="upper right")
ax2.legend(loc="upper right")

ax1.set_ylabel("Signal (millivolt)")
ax2.set_ylabel("Signal (millivolt)")

ax1.plot(data_interval, rms, "k-")
ax2.plot(data_interval, rms, "k-")

plt.show()

rmse = np.mean(np.fromiter([(y_test[i] - rms[int((x_test[i])*500)])**2 for i in range(len(x_test))], dtype=float))

rmse = np.sqrt(rmse)
print("Running Mean Smoother => RMSE is", rmse, "when h is", h)



# %%
# Kernel Smoother
h = 0.02

def my_K(u):
    return (1 / np.sqrt(2 * np.pi) * np.exp(-(u*u)/2))

predictions = np.zeros(len(data_interval))
for i in range(len(data_interval)):
  predictions[i] =np.sum(my_K((data_interval[i] - x_train)/h)*y_train) / np.sum(my_K((data_interval[i]-x_train)/h))


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))

ax1.plot(x_train, y_train, "b.", markersize = 10, label = "Training")
ax2.plot(x_test, y_test, "r.", markersize = 10, label = "Test")

ax1.set_xlabel("Time (sec)")
ax2.set_xlabel("Time (sec)")
ax1.legend(loc="upper right")
ax2.legend(loc="upper right")

ax1.set_ylabel("Signal (millivolt)")
ax2.set_ylabel("Signal (millivolt)")


ax1.plot(data_interval, predictions, "k-")
ax2.plot(data_interval, predictions, "k-")

plt.show()

rmse = [(y_test[i] - predictions[int((x_test[i])*500)])**2 for i in range(len(x_test))]
rmse = np.sum(rmse)/np.shape(x_test)[0]
rmse = np.sqrt(rmse)

print("Kernel Smoother => RMSE is", rmse,"when h is", h)




