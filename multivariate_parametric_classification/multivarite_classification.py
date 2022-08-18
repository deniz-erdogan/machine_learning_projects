# Written by Deniz Erdogan
# Multivariate Parametric Classification


# ## Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import math
import scipy.stats as stats
import pandas as pd

# ## Parameters for Generating The Data
# mean parameters
mean1 = np.array([0.0, 4.5])
mean2 = np.array([-4.5, -1.0])
mean3 = np.array([4.5, -1.0])
mean4 = np.array([0.0, -4.0])
class_means = np.stack((mean1,mean2,mean3,mean4), axis=0)

# standard deviation parameters
def build_covariance_matrix(el11, el12, el22):
    return np.stack((np.array([el11, el12]), np.array([el12, el22])), axis=0)


covariance1 = build_covariance_matrix(3.2, 0.0, 1.2)
covariance2 = build_covariance_matrix(1.2, 0.8, 1.2)
covariance3 = build_covariance_matrix(1.2, -0.8, 1.2)
covariance4 = build_covariance_matrix(1.2, 0, 3.2)
class_covariances = np.stack((covariance1, covariance2, covariance3, covariance4), axis=0)

class_sizes = np.array([105, 145, 135, 115])
N = np.sum(class_sizes)
K = 4


# ## Generating Data
points1 = np.random.multivariate_normal(class_means[0], class_covariances[0], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1], class_covariances[1], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2], class_covariances[2], class_sizes[2])
points4 = np.random.multivariate_normal(class_means[3], class_covariances[3], class_sizes[3])
points = np.concatenate((points1, points2, points3, points4))
print(np.shape(points1[0]))

y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2]), np.repeat(4, class_sizes[3]))) 


# ## Plotting the Data
plt.figure(figsize=(10, 10))
plt.plot(points1[:, 0], points1[:, 1], "r.", markersize=10)
plt.plot(points2[:, 0], points2[:, 1], "g.", markersize=10)
plt.plot(points3[:, 0], points3[:, 1], "b.", markersize=10)
plt.plot(points4[:, 0], points4[:, 1], "m.", markersize=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# ## Exporting Data
# write data to a file
np.savetxt("hw1_data_set.csv", np.stack((points[:,0], points[:,1], y), axis = 1), fmt = "%f,%f,%d")


# ## Parameter Estimation
class_priors = [np.mean(y == (c + 1)) for c in range(K)]
print(class_priors)

sample_means1 = [np.mean(points1[:, 0]), np.mean(points1[:, 1])]
sample_means2 = [np.mean(points2[:, 0]), np.mean(points2[:, 1])]
sample_means3 = [np.mean(points3[:, 0]), np.mean(points3[:, 1])]
sample_means4 = [np.mean(points4[:, 0]), np.mean(points4[:, 1])]
sample_means = np.stack((sample_means1, sample_means2, sample_means3, sample_means4), axis=0)
print(sample_means)

sample_covs = np.stack(([np.cov(points[y == (c+1)][:,0], points[y == (c+1)][:,1]) for c in range(K)]), axis=0)
print(sample_covs)


# ## Calculate Wc's and wc's of classes
Wc = np.stack(([-0.5 * linalg.inv(sample_covs[c]) for c in range(K)]),axis=0)
print(np.shape(Wc))

wc = np.stack(([np.matmul(linalg.inv(sample_covs[c]),sample_means[c]) for c in range(K)]),axis=0)

wc0 = np.stack(([(-0.5) * np.matmul(np.matmul(np.transpose(sample_means[c]), np.linalg.inv(sample_covs[c])), sample_means[c]) 
            + (-0.5) * N * np.log(2 * math.pi) 
            + (-0.5) * np.log(np.linalg.det(sample_covs[c])) + np.log(class_priors[c])
                for c in range(K)]), axis=0)


g = np.stack(([np.stack(([ np.matmul(np.matmul(point, Wc[c]), np.transpose(point)) 
        + np.matmul(np.transpose(wc[c]), np.transpose(point)) + wc0[c] 
        for point in points]),axis=0)
                for c in range(K)]), axis=0)

A = []
for i in range(0, N):
    assignment = 0
    if max(g[0][i], g[1][i], g[2][i], g[3][i]) == g[0][i]:
        assignment = 1
    if max(g[0][i], g[1][i], g[2][i], g[3][i]) == g[1][i]:
        assignment = 2
    if max(g[0][i], g[1][i], g[2][i], g[3][i]) == g[2][i]:
        assignment = 3
    if max(g[0][i], g[1][i], g[2][i], g[3][i]) == g[3][i]:
        assignment = 4
    A.append(assignment)


plt.figure(figsize=(10, 10))
plt.plot(points[y == 1, 0], points[y == 1, 1], "r.", markersize=10)
plt.plot(points[y == 2, 0], points[y == 2, 1], "g.", markersize=10)
plt.plot(points[y == 3, 0], points[y == 3, 1], "b.", markersize=10)
plt.plot(points[y == 4, 0], points[y == 4, 1], "m.", markersize=10)
plt.plot(points[A != y, 0], points[A != y, 1], "ko", markersize=12, fillstyle="none")
plt.show()


confusion_matrix = pd.crosstab(np.transpose(A), y, rownames=['y_pred'], colnames=['y_truth'])
print(confusion_matrix)


# ## Generating the Grid
x1vals = np.linspace(-8, +8, 321)
x2vals = x1vals.copy()

x1vals, x2vals = np.meshgrid(x1vals, x2vals)
grid_points = np.transpose(np.vstack((x1vals, x2vals)))

grid_scores = np.stack(([ x1vals * Wc[c][0, 0] * x1vals + x1vals * Wc[c][0, 1] * x2vals + x2vals * Wc[c][
    1, 0] * x1vals + x2vals * Wc[c][1, 1] * x2vals + wc[c][0] * x1vals + wc[c][1] * x2vals + wc0[c]
                for c in range(K)]), axis=0)

print(np.shape(grid_scores))
grid_assignments = np.zeros((321,321))
temp = 0
for i in range(321):
    for j in range(321):
        if max(grid_scores[0][i][j], grid_scores[1][i][j], grid_scores[2][i][j], grid_scores[3][i][j]) == grid_scores[0][i][j]:
            temp = 1
        if max(grid_scores[0][i][j], grid_scores[1][i][j], grid_scores[2][i][j], grid_scores[3][i][j]) == grid_scores[1][i][j]:
            temp = 2
        if max(grid_scores[0][i][j], grid_scores[1][i][j], grid_scores[2][i][j], grid_scores[3][i][j]) == grid_scores[2][i][j]:
            temp = 3
        if max(grid_scores[0][i][j], grid_scores[1][i][j], grid_scores[2][i][j], grid_scores[3][i][j]) == grid_scores[3][i][j]:
            temp = 4
        grid_assignments[i,j] = temp
print(np.shape(grid_assignments))


plt.figure(figsize=(10, 10))
plt.scatter(x1vals[grid_assignments == 1], x2vals[grid_assignments == 1], c="r", alpha = 0.1, s = 2)
plt.scatter(x1vals[grid_assignments == 2], x2vals[grid_assignments == 2], c="g", alpha = 0.1, s = 2)
plt.scatter(x1vals[grid_assignments == 3], x2vals[grid_assignments == 3], c="b", alpha = 0.1, s = 2)
plt.scatter(x1vals[grid_assignments == 4], x2vals[grid_assignments == 4], c="m", alpha = 0.1, s = 2)

plt.plot(points[y == 1, 0], points[y == 1, 1], "r.", markersize=10)
plt.plot(points[y == 2, 0], points[y == 2, 1], "g.", markersize=10)
plt.plot(points[y == 3, 0], points[y == 3, 1], "b.", markersize=10)
plt.plot(points[y == 4, 0], points[y == 4, 1], "m.", markersize=10)
plt.plot(points[A != y, 0], points[A != y, 1], "ko", markersize=12, fillstyle="none")
plt.show()
