'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
import time

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z)))
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #
    n = training_data.shape[0]

    training_bias = np.concatenate((np.full((n, 1), 1), training_data), axis=1)
    # w1output = sigmoid(training_bias @ (w1.T))
    w1output = sigmoid(np.matmul(training_bias, (w1.T)))
    bias = np.concatenate((np.full((w1output.shape[0], 1), 1), w1output), axis=1)
    # w2output = sigmoid(bias @ (w2.T))
    w2output = sigmoid(np.matmul(bias, (w2.T)))

    gt = np.full((n, n_class), 0)

    for i in range(n):
        gt[i][training_label[i]] = 1
    pgt = (1.0 - gt)
    pw2out = (1.0 - w2output)

    lw2out = np.log(w2output)
    lpw2out = np.log(pw2out)

    error = (np.sum(np.multiply(gt, lw2out) + np.multiply(pgt, lpw2out))) / ((-1.0) * n)

    # gradient calculation
    delta = w2output - gt
    grad_w2 = np.matmul(delta.T, bias)
    grad_w1 = np.matmul(((np.matmul(delta, w2)) * (bias * (1 - bias))).T, training_bias)[1:, :]

    # regularization
    regularization = lambdaval * (np.sum(w1 ** 2, dtype=np.float32) + np.sum(w2 ** 2, dtype=np.float32)) / (2 * n)
    obj_val = error + regularization

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate(
        (((grad_w1 + (lambdaval * w1)) / n).flatten(), ((grad_w2 + (lambdaval * w2)) / n).flatten()), 0)

    return (obj_val, obj_grad)


# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    bias_layer_1 = np.full((data.shape[0], 1), 1)
    training_data = np.hstack((bias_layer_1, data))

    # input_layer -> hidden_layer
    aj = np.matmul(training_data, np.transpose(w1))
    zj = sigmoid(aj)

    bias_layer_2 = np.full((zj.shape[0], 1), 1)
    zj = np.hstack((bias_layer_2, zj))

    # hidden_layer -> output_layer
    bl = np.matmul(zj, np.transpose(w2))
    ol = sigmoid(bl)

    labels = np.argmax(ol, axis=1)

    return labels


# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
c=0
time_arr = np.zeros(len(range(4,28,4)))
for x in range(4,28,4):
    n_hidden = x
    print("Number of Hidden Nodes :" + str(n_hidden))
    # set the number of nodes in output unit
    n_class = 2

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden);
    initial_w2 = initializeWeights(n_hidden, n_class);
    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
    # set the regularization hyper-parameter
    count=0
    acc_arr=np.zeros(len(range(0,70,10)))
    for y in range(0,70,10):
        lambdaval = y
        print("Lambda value:" + str(lambdaval))
        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

        # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
        time1 = time.time()
        opts = {'maxiter': 50}  # Preferred value.

        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        params = nn_params.get('x')
        time2 = time.time()
        # Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        # Test the computed parameters
        predicted_label = nnPredict(w1, w2, train_data)
        # find the accuracy on Training Dataset
        print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
        predicted_label = nnPredict(w1, w2, validation_data)
        # find the accuracy on Validation Dataset
        validation_accuracy = 100 * np.mean((predicted_label == validation_label).astype(float))
        acc_arr[count] = validation_accuracy
        count = count + 1
        print('\n Validation set Accuracy:' + str(validation_accuracy) + '%')
        predicted_label = nnPredict(w1, w2, test_data)
        # find the accuracy on Validation Dataset
        print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

    time_arr[c] = time2 - time1
    print("Training Time:" + str(time_arr[c]) + " seconds")
    c = c + 1
    
    fig = plt.figure(figsize=[6, 6])
    # plt.subplot(1, 2, 1)
    plt.plot(range(0,70,10), acc_arr)
    plt.title('Lambda Value v/s Accuracy')
    plt.savefig("LambdaValueVsAccuracyfor" + str(x) + "hiddenlayers" + ".jpg")
    # plt.show()



fig = plt.figure(figsize=[6, 6])
# plt.subplot(1, 2, 1)
plt.plot(range(4,28,4), time_arr)
plt.title('Number of Hidden Nodes v/s Training Time')
plt.savefig("HiddenLayersVsTrainTime" + ".jpg")
# plt.show()
