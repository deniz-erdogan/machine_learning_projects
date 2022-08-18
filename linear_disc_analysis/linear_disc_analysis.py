
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd


# read data into memory
images = np.genfromtxt("hw07_data_set_images.csv", delimiter = ",")
labels = np.genfromtxt("hw07_data_set_labels.csv", delimiter = ",").astype(int)

# get X and y values
X_train = images[:2000,:]
y_train = labels[:2000]

X_test = images[2000:,:]
y_test = labels[2000:]

# get number of samples and number of features
N = X_train.shape[0]
D = X_train.shape[1]
K = np.max(y_train)


# obtaining the sample means
means = np.asarray([np.mean(X_train[y_train == (c + 1)],axis=0) for c in range(K)])
sample_mean = np.mean(X_train, axis = 0)

Sw = np.zeros((D, D))
for i in range(N):
    for c in range(K):
        if(y_train[i] == c+1):
            # for some reason, np.matmul did not seem to work for me
            Sw += np.outer((X_train[i,:]-means[c]), (X_train[i,:]-means[c]))
            # Sw += np.matmul((X_train[i,:]-means[c]), np.transpose((X_train[i,:]-means[c])))


Sb = np.zeros((D, D))
for i in range(N):
    for c in range(K):
        if(y_train[i] == (c+1)):
            Sb += np.outer((means[c,:]-sample_mean), (means[c,:]-sample_mean))

print("Printing the matrices: (assuming hw desc. printed Sw[0:4, 0:4])")
print("Sw:")
print(Sw[0:4, 0:4])
print("Sb:")
print(Sb[0:4, 0:4])


SwSb = np.matmul(np.linalg.inv(Sw),Sb)
values, vectors = linalg.eig(SwSb)
values = np.real(values)
vectors = np.real(vectors)
print("First 9 eigenvalues of Sw^1Sb matrix is follows:")
print(values[0:9])



point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])
Z_train = np.matmul(X_train - np.mean(X_train, axis = 0), vectors[:,[0, 1]])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

ax1.set_title("Training Data")
ax1.set_xlabel("Component 1")
ax1.set_ylabel("Component 2")
ax1.set_xlim([-6,6])
ax1.set_ylim([-6,6])

for c in range(K):
    ax1.plot(Z_train[y_train == c + 1, 0], Z_train[y_train == c + 1, 1], marker = "o", markersize = 2, linestyle = "none", color = point_colors[c])

ax1.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"],
           loc = "upper left", markerscale = 2, fontsize = 8)


Z_test = np.matmul(X_test - np.mean(X_test, axis = 0), vectors[:,[0, 1]])

ax2.set_title("Test Data")
ax2.set_xlabel("Component 1")
ax2.set_ylabel("Component 2")
ax2.set_xlim([-6,6])
ax2.set_ylim([-6,6])

for c in range(K):
    ax2.plot(Z_test[y_test == c + 1, 0], Z_test[y_test == c + 1, 1], marker = "o", markersize = 2, linestyle = "none", color = point_colors[c])
    
ax2.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"],
           loc = "upper left", markerscale = 2, fontsize = 8)

plt.show()


Z_train = np.matmul(X_train - sample_mean, vectors[:,0:9])
Z_test = np.matmul(X_test - sample_mean, vectors[:,0:9])

k = 11
training_scores = np.zeros((N, K))
for c in range(K):
    training_scores[:,c] = np.asarray([np.sum(y_train[np.argsort(np.linalg.norm(z - Z_train, axis=1))[range(k)]] == c + 1) for z in Z_train]) / K


training_predictions = np.argmax(training_scores, axis=1) + 1
confusion_matrix = pd.crosstab(training_predictions, y_train, rownames = ["y_train"], colnames = ["y_hat"])
print("Confusion Matrix for the training set:")
print(confusion_matrix)


test_scores = np.zeros((N, K))
for c in range(K):
    test_scores[:,c] = np.asarray([np.sum(y_train[np.argsort(np.linalg.norm(z - Z_train, axis=1))[range(k)]] == c + 1) for z in Z_test]) / K


test_predictions = np.argmax(test_scores, axis=1) + 1
confusion_matrix = pd.crosstab(test_predictions, y_test, rownames = ["y_test"], colnames = ["y_hat"])
print("Confusion Matrix for the test set:")
print(confusion_matrix)




