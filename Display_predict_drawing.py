import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#data = np.load('matrices.npz')

#this file just gets a random examples from the mnist database then compares the answer 
data = np.load('MNIST\Github\weights_baises.npz')
W1, b1, W2, b2, W3, b3 = data['W1'], data['b1'], data['W2'], data['b2'], data['W3'], data['b3']

data = pd.read_csv(r'\MNIST\development\train.csv')

data = np.array(data)

m,n = data.shape

np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] 
X_dev = (X_dev / 255.) - 1 

data_train = data[1000:m].T#
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = (X_train / 255.) - 0.5 





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




def get_predictions(A3):
    return np.argmax(A3, 0)

def plot_random_number(one_example_X, one_example_Y, prediction):
    #this works with data where each column is one example
    #label = data[0,column_index]
    #image_array = data[1:,column_index]
    image_array = one_example_X
    label = one_example_Y
    guess = prediction
    # Reshape the array into a 28x28 image
    image = image_array.reshape(28, 28)
    inverted_image = 255 - image

    # Create a scatter plot of the inverted image
    x, y = np.meshgrid(range(28), range(28))
    plt.scatter(x, 27 - y, c=inverted_image, cmap='gray', marker='s')

    # Set labels and title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Answer: {label}  Guess: {guess}")

    # Show the plot
    plt.show()
    
    plt.close('all')

def predict_single_example(index_number, W1, b1, W2, b2, W3, b3):
    x = X_train[:,index_number]
    print(x)
    y = Y_train[index_number]
    # Ensure the input is a 2D array of shape (1, 784)
    x = np.reshape(x, (1, -1))

    # Perform forward propagation
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, x.T) # note that x is transposed
    
    # Get the prediction by selecting the index of the maximum value in A3
    prediction = np.argmax(A3)
    print("Answer: ", y)
    print("Prediction: ", prediction)
    plot_random_number(x, y, prediction)
    return prediction  # Return the single prediction value

random_index= np.random.randint(1,100)
print(random_index)
predict_single_example(random_index, W1, b1, W2, b2, W3, b3)