import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import scipy

# read data into memory
X = np.genfromtxt("hw09_data_set.csv", delimiter = ",")

N = X.shape[0]
K = 9


initial_centroids = [242, 528, 570, 590, 648, 667, 774, 891, 955]
def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        centroids = X[initial_centroids]
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)

# %%
B = spa.distance_matrix(X, X) <= 2
for i in range(N):
    B[i,i] = 0


plt.figure(figsize = (8, 8))
plt.plot(X[:,0], X[:,1], "k.", markersize = 10)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
for i in range(N):
    for j in range(i):
        if B[i][j] == 1:
            plt.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], color = "#808080", linewidth = 0.3)

plt.plot(X[:,0], X[:,1], "k.", markersize = 10)
plt.show()


# D and L matrices
D = np.eye(N) * np.sum(B, axis=0)


L = np.eye(N) - np.matmul(np.matmul(scipy.linalg.fractional_matrix_power(D, -0.5), B), scipy.linalg.fractional_matrix_power(D, -0.5))




R = 5

values, vectors = np.linalg.eig(L)
value_indexes = np.argsort(values)
vectors = vectors[:, value_indexes]
Z = vectors[:,1:R+1]

# print(vectors[0])
# print(vectors[1])
# print(vectors[2])
print(Z.shape)


K = 9

def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12, 
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

plt.figure(figsize = (12, 12))

centroids = None
memberships = None
iteration = 1
while True:
    print("Iteration#{}:".format(iteration))

    old_centroids = centroids
    centroids = update_centroids(memberships, Z)
    if np.alltrue(centroids == old_centroids):
        break

    old_memberships = memberships
    memberships = update_memberships(centroids, Z)
    if np.alltrue(memberships == old_memberships):
        break
    iteration = iteration + 1

centroids = update_centroids(memberships, X)
plot_current_state(centroids, memberships, X)
