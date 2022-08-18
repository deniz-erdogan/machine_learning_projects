import matplotlib.pyplot as plt
import numpy as np


# Import Data
data_set = np.genfromtxt("hw05_data_set_train.csv", delimiter = ",")
test_set = np.genfromtxt("hw05_data_set_test.csv", delimiter = ",")

x_train = data_set[:,0]
y_train= data_set[:,1]

x_test = test_set[:,0]
y_test= test_set[:,1]


def learn_decision_tree_regression(P, x_train, y_train):
    
  node_indices = {}
  is_terminal = {}
  need_split = {}
  node_splits = {}
  
  # put all training instances into the root node
  node_indices[1] = np.array(range(x_train.shape[0]))
  is_terminal[1] = False
  need_split[1] = True
  
  while True:
      
    split_nodes = [key for key, value in need_split.items() if value == True]
   
    # check whether we reach all terminal nodes
    if len(split_nodes) == 0:
      break

    # find best split positions for all nodes
    for split_node in split_nodes:
      data_indices = node_indices[split_node]
      need_split[split_node] = False
      
      if len(data_indices) <= P:
        node_splits[split_node] = np.mean(y_train[data_indices])
        is_terminal[split_node] = True
      else:
        is_terminal[split_node] = False
        
        uniques = np.sort(np.unique(x_train[data_indices]))
        split_positions = (uniques[1:len(uniques)] + uniques[0:(len(uniques) - 1)]) / 2
        split_scores = np.repeat(0.0, split_positions.shape[0])
        
        for split in range(len(split_positions)):
          left_indices = data_indices[x_train[data_indices] <= split_positions[split]]
          right_indices = data_indices[x_train[data_indices] > split_positions[split]]
          
          left_child_avg = np.mean(y_train[left_indices])
          right_child_avg = np.mean(y_train[right_indices])

          error = np.sum((y_train[left_indices]-left_child_avg)**2) + np.sum((y_train[right_indices] - right_child_avg) ** 2)
          split_scores[split] = error / (len(left_indices) + len(right_indices))
          
        best_splits = split_positions[np.argmin(split_scores)]
        node_splits[split_node] = best_splits

        
        left_indices = data_indices[x_train[data_indices] <= best_splits]
        node_indices[2 * split_node] = left_indices
        is_terminal[2 * split_node] = False
        need_split[2 * split_node] = True
        
        right_indices = data_indices[x_train[data_indices] > best_splits]
        node_indices[2 * split_node + 1] = right_indices
        is_terminal[2 * split_node + 1] = False
        need_split[2 * split_node + 1] = True
        
  return is_terminal, node_splits

def predict_values(data, node_splits, is_terminal):
    predicted_values = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        leaf = 1
        while True:
            if is_terminal[leaf] == True:
                predicted_values[i] = node_splits[leaf]
                break
            else:
                if data[i] <= node_splits[leaf]:
                    leaf *= 2
                else:
                    leaf *= 2
                    leaf += 1
    return predicted_values

def calculate_error(y_truth, y_hat):
    return np.sqrt(np.mean((y_truth - y_hat)**2))

is_terminal, node_splits = learn_decision_tree_regression(30, x_train, y_train)

data_interval = np.linspace(0, 2, 1001)
predicted_values = predict_values(data_interval, node_splits, is_terminal)



# Plotting the data with our regression results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))

ax1.plot(x_train, y_train, "b.", markersize = 10, label="Training")
ax2.plot(x_test, y_test, "r.", markersize = 10, label="Test")

ax1.plot(data_interval, predicted_values, "k-")
ax2.plot(data_interval, predicted_values, "k-")

ax1.set_xlabel("Time (sec)")
ax2.set_xlabel("Time (sec)")
ax1.legend(loc="upper right")
ax2.legend(loc="upper right")

ax1.set_ylabel("Signal (millivolt)")
ax2.set_ylabel("Signal (millivolt)")

plt.show()


# Calculating the error on both sets
training_predictions = predict_values(x_train, node_splits, is_terminal)
error = calculate_error(y_train, training_predictions)
print("RMSE on training set is", error, "when P is 30.")

test_predictions = predict_values(x_test, node_splits, is_terminal)
error = calculate_error(y_test, test_predictions)
print("RMSE on test set is", error, "when P is 30.")


pre_pruning = np.linspace(10, 50, num=9)
training_error = np.repeat(0.0, np.shape(pre_pruning)[0])
test_error = np.repeat(0.0, np.shape(pre_pruning)[0])

for i in range(pre_pruning.shape[0]):
    is_terminal_iter, node_splits_iter = learn_decision_tree_regression(pre_pruning[i], x_train, y_train)
    training_error[i] = calculate_error(y_train, predict_values(x_train, node_splits_iter, is_terminal_iter))

    test_error[i] = calculate_error(y_test, predict_values(x_test, node_splits_iter, is_terminal_iter))

plt.figure(figsize = (10, 5))
plt.plot(pre_pruning, training_error, "bo-", linewidth=2, markersize = 10, label="training")
plt.plot(pre_pruning, test_error, "ro-", linewidth=2, markersize = 10, label="test")
plt.legend()
plt.xlabel("Pre Pruning Size (P)")
plt.ylabel("RMSE")

plt.show()



