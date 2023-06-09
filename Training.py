import numpy as np
import pandas as pd

#this is a database where each row is an exmaple and each column is the label then pixel values.
data = pd.read_csv(r'\MNIST\development\train.csv')

data = np.array(data)
m,n = data.shape

#seed to standardize testing
np.random.seed(0)
#shuffle test data
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = (X_dev / 255.) - 0.5

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = (X_train / 255.) - 0.5

output_nodes = 10
input_nodes = 784

#weights and biases initialization 
def init_params(nodes_layer_1, nodes_layer_2):
    W1 = np.random.randn(nodes_layer_1, input_nodes) * np.sqrt(2./input_nodes)
    b1 = np.zeros((nodes_layer_1, 1))

    W2 = np.random.randn(nodes_layer_2, nodes_layer_1) * np.sqrt(2./nodes_layer_1)
    b2 = np.zeros((nodes_layer_2, 1))

    W3 = np.random.randn(output_nodes ,nodes_layer_2) * np.sqrt(2./nodes_layer_2)
    b3 = np.zeros((output_nodes, 1))

    return W1,b1,W2,b2,W3,b3


#forward prop
def ReLU(Z):
    return np.maximum(Z,0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3


#backprop
def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    
    m = X.shape[1]  # Number of examples
    
    # Convert Y to one-hot vectors
    one_hot_Y = one_hot(Y)
    
    # Compute the derivative of the cost w.r.t Z3
    dZ3 = A3 - one_hot_Y
    
    # Compute the gradients for W3 and b3
    dW3 = (1/m) * np.dot(dZ3, A2.T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
    
    # Compute the derivative of the cost w.r.t Z2
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * ReLU_deriv(Z2)
    
    # Compute the gradients for W2 and b2
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # Compute the derivative of the cost w.r.t Z1
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * ReLU_deriv(Z1)
    
    # Compute the gradients for W1 and b1
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    W3 = W3 - alpha * dW3  
    b3 = b3 - alpha * db3    
    return W1, b1, W2, b2, W3, b3

#Gradient Descent 
def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    print("Guesses:      Answers:")
    print(predictions[0:5], Y[0:5])
    return np.sum(predictions == Y) / Y.size
def gradient_descent(X, Y, alpha, iterations, nodes_layer_1, nodes_layer_2):

    W1,b1,W2,b2,W3,b3 = init_params(nodes_layer_1, nodes_layer_2)
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3  = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 25 == 0:
                    predictions = get_predictions(A3)
                    accuracy = get_accuracy(predictions, Y)
                    print("Iteration: ", i)
                    print("Accuracy: ", accuracy)
    info = [accuracy, alpha, nodes_layer_1, nodes_layer_2, iterations]
    analysis(accuracy, alpha, nodes_layer_1, nodes_layer_2, iterations)
    return W1,b1,W2,b2,W3,b3,info

#gives info on each gradient descent
def analysis(accuracy, alpha, nodes_layer_1, nodes_layer_2, iterations):
    print("\n\nFinal Accuracy: ", accuracy)
    print("Learning Rate: ",  alpha)
    print("Number of L1 nodes: ",  nodes_layer_1, "  Number of L2 nodes: ",  nodes_layer_2)
    print("Total Iterations: ",  iterations)

#tests the network on the test set
def make_predictions(X,Y, W1,b1,W2,b2,W3,b3, info):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    accuracy = get_accuracy(predictions, Y)
    diff_in_accuracy = info[0] - accuracy
    return accuracy, diff_in_accuracy

#compare the different ouputs of the different configurations
def find_best_network():
        W1,b1,W2,b2,W3,b3,info = gradient_descent(X_train, Y_train, 0.01, 500, 30, 30)
        test_accuracy, difference = make_predictions(X_dev, Y_dev, W1,b1,W2,b2,W3,b3,info)
        print("\nTraining Accuracy: ",  np.round(info[0], decimals=2), "\nTest Set Accuracy: ",  np.round(test_accuracy, decimals=2))
        print("Difference: ", np.round(difference, decimals=2))
        print("Learning rate: ", info[1], "\nNodes in both layers: ", info[2])
        W1,b1,W2,b2,W3,b3,info = gradient_descent(X_train, Y_train, 0.1, 500, 30, 30)
        test_accuracy, difference = make_predictions(X_dev, Y_dev, W1,b1,W2,b2,W3,b3,info)
        print("\nTraining Accuracy: ",  np.round(info[0], decimals=2), "\nTest Set Accuracy: ",  np.round(test_accuracy, decimals=2))
        print("Difference: ", np.round(difference, decimals=2))
        print("Learning rate: ", info[1], "\nNodes in both layers: ", info[2])
        return W1,b1,W2,b2,W3,b3

W1,b1,W2,b2,W3,b3 = find_best_network()

#save to another file to be used later on 
#np.savez('matrices.npz', W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)