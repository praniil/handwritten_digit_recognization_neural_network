import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/home/pranil/python_projects/digit_recog_neural_network/dataset/train.csv')
# print(data.head(20))

data = np.array(data)
#rows, column
m, n = data.shape
#rows are shuffled
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev/ 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.    #normalize the pixel intensity values in images range[0, 255]
_, m_train = X_train.shape

#initiaization of neural network parameter
def initialization_params():
    W1 = np.random.rand(10, 784) - 0.5  #2D array with 10 rows and 784 columns [0,1) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


#activation function inner layer
def ReLU(z):
    return np.maximum(z, 0)

#activation function output layer
def softmax(z):
    A = np.exp(z) / sum(np.exp(z))
    return A
