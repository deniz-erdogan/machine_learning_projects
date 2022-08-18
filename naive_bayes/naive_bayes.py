# Deniz Erdogan
# 0069572
# Homework 2
# Naive Bayes Classifier


import math 
import numpy as np
import pandas as pd


# Defining our safelog
def safelog(x):
  return np.log(x + 1e-100)

# Load the Data
images = np.loadtxt("hw2_data_set_images.csv",delimiter = ",")
y = np.loadtxt('hw2_data_set_labels.csv', delimiter = ",")
# print(np.shape(images))

K = max(y) # number of classes

training_images = np.stack(([images[i*39 : i*39+25] for i in range(int(K))]), axis=0)
# print(np.shape(training_images))

test_images = np.stack(([images[i*39+25 : i*39+39] for i in range(int(K))]), axis=0)
# print(np.shape(test_images))

training_labels = np.stack(([y[i*39 : i*39+25] for i in range(int(K))]), axis=0)
# print(np.shape(training_labels))

test_labels = np.stack(([y[i*39+25 : i*39+39] for i in range(int(K))]), axis=0)
# print(np.shape(test_labels))




# Parameter Estimation

# Class priors
N = np.size(training_labels)
class_priors = np.stack(([len(training_labels[i])/N for i in range(int(K))]), axis=0)
print("Priors:")
print(class_priors)

# Sample means
sample_means = [np.mean(training_images[i], axis=0) for i in range(int(K))]
print("Sample means:")
print(sample_means)

# Score Values
training_images  = training_images.reshape(125,320)
scores = np.stack(([ np.stack( ([np.sum(training_images[d] * safelog(sample_means[c]) +
                        (1 - training_images[d]) * safelog(1 - sample_means[c])) + safelog(class_priors[c]) 
                        for d in range(np.shape(training_images)[0])]), axis=0)
        for c in range(int(K))]),axis=0)
# print(np.shape(scores))




assignments = []
for i in range(125):
  temp = 0
  if (max(scores[0][i], scores[1][i], scores[2][i],scores[3][i],scores[4][i] ) == scores[0][i]): 
      temp = 1
  if (max(scores[0][i], scores[1][i], scores[2][i],scores[3][i],scores[4][i] ) == scores[1][i]):
      temp = 2
  if (max(scores[0][i], scores[1][i], scores[2][i],scores[3][i],scores[4][i] ) == scores[2][i]):
      temp = 3
  if (max(scores[0][i], scores[1][i], scores[2][i],scores[3][i],scores[4][i] ) == scores[3][i]):
      temp = 4
  if (max(scores[0][i], scores[1][i], scores[2][i],scores[3][i],scores[4][i] ) == scores[4][i]):
      temp = 5
  assignments.append(temp)

print("Training set:")
confusion_matrix = pd.crosstab(np.transpose(assignments), training_labels.flatten(), rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# Now for test data
test_images  = test_images.reshape(5*14,320)
test_scores = np.stack(([ np.stack( ([np.sum(test_images[d] * safelog(sample_means[c]) +
                        (1 - test_images[d]) * safelog(1 - sample_means[c])) + safelog(class_priors[c]) 
                        for d in range(np.shape(test_images)[0])]), axis=0)
        for c in range(int(K))]),axis=0)

assignments = []
for i in range(70):
  temp = 0
  if (max(test_scores[0][i], test_scores[1][i], test_scores[2][i],test_scores[3][i],test_scores[4][i] ) == test_scores[0][i]): 
      temp = 1
  if (max(test_scores[0][i], test_scores[1][i], test_scores[2][i],test_scores[3][i],test_scores[4][i] ) == test_scores[1][i]):
      temp = 2
  if (max(test_scores[0][i], test_scores[1][i], test_scores[2][i],test_scores[3][i],test_scores[4][i] ) == test_scores[2][i]):
      temp = 3
  if (max(test_scores[0][i], test_scores[1][i], test_scores[2][i],test_scores[3][i],test_scores[4][i] ) == test_scores[3][i]):
      temp = 4
  if (max(test_scores[0][i], test_scores[1][i], test_scores[2][i],test_scores[3][i],test_scores[4][i] ) == test_scores[4][i]):
      temp = 5
  assignments.append(temp)


print("Test Set:")
test_confusion_matrix = pd.crosstab(np.transpose(assignments), test_labels.flatten(), rownames = ['y_pred'], colnames = ['y_truth'])
print(test_confusion_matrix)



