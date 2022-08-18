import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import scipy.stats

data_points = np.genfromtxt("hw08_data_set.csv", delimiter = ",")
centroids = np.genfromtxt("hw08_initial_centroids.csv", delimiter = ",")


K = 9

def assign_to_nearest_center(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)


memberships = assign_to_nearest_center(centroids, data_points)

covariances = np.stack(([np.cov(data_points[memberships==k][:,0], data_points[memberships==k][:,1]) for k in range(K)]), axis=0) 
priors = np.stack(([np.mean(memberships == k) for k in range(K)]), axis = 0)
means_x = np.stack(([np.mean(data_points[memberships==k][:,0]) for k in range(K)]), axis=0)
means_y = np.stack(([np.mean(data_points[memberships==k][:,1]) for k in range(K)]), axis=0)

means = np.column_stack((means_x, means_y))

print(means)





def update_memberships(means, covariances, X):
    D = np.stack(([scipy.stats.multivariate_normal.pdf(X, mean=means[k], cov=covariances[k]) * priors[k] for k in range(K)]),axis=0)
    memberships = np.argmax(D, axis = 0)
    return(memberships)


for _ in range(100):
    memberships = update_memberships(means, covariances, data_points)

    priors = np.stack(([np.mean(memberships == k) for k in range(K)]), axis = 0)
    means_x = np.stack(([np.mean(data_points[memberships==k][:,0]) for k in range(K)]), axis=0)
    means_y = np.stack(([np.mean(data_points[memberships==k][:,1]) for k in range(K)]), axis=0)

    means = np.column_stack((means_x, means_y))
    covariances = np.stack(([np.cov(data_points[memberships==k][:,0], data_points[memberships==k][:,1]) for k in range(K)]), axis=0) 






print(means)


def plot_current_state(memberships, X):

    # mean parameters
    class_means = np.array([[5,-5,-5,5,5,0,-5,0,0],[5,5,-5,-5,0,5,0,-5,0]]).T
    # covariance parameters
    class_covariances = np.array([[[0.8,-0.6],[-0.6,0.8]]
                                ,[[0.8,0.6],[0.6,0.8]]
                                ,[[0.8,-0.6],[-0.6,0.8]]
                                ,[[0.8,0.6],[0.6,0.8]]
                                ,[[0.2,0],[0,1.2]]
                                ,[[1.2,0],[0,0.2]]
                                ,[[0.2,0],[0,1.2]]
                                ,[[1.2,0],[0,0.2]]
                                ,[[1.6,0],[0,1.5]]])

    plt.figure(figsize = (8, 8))
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])

    for c in range(K):
        plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
        
    x1_interval = np.linspace(-8, +8, 1001)
    x2_interval = np.linspace(-8, +8, 1001)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T

    for c in range(K):
        D1 = scipy.stats.multivariate_normal.pdf(X_grid, mean = means[c,:],
                                    cov =covariances[c,:,:])
        D1 = D1.reshape((len(x1_interval), len(x2_interval)))
        plt.contour(x1_grid, x2_grid, D1, levels = [0.05],
                colors = cluster_colors[c], linestyles = "solid")

        D2 = scipy.stats.multivariate_normal.pdf(X_grid, mean = class_means[c,:],
                                       cov = class_covariances[c,:,:])
        D2 = D2.reshape((len(x1_interval), len(x2_interval)))

        plt.contour(x1_grid, x2_grid, D2, levels = [0.05],
                colors = "k", linestyles = "dashed")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

plot_current_state(memberships, data_points)



