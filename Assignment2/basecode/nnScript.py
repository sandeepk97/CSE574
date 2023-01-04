import pickle
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    Z = 1.0 / (1.0 + np.exp(-1.0 * z))
    return  Z


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))

#     train_preprocess = np.zeros(shape=(3000, 784))
#     validation_preprocess = np.zeros(shape=(1000, 784))
#     test_preprocess = np.zeros(shape=(1000, 784))
#     train_label_preprocess = np.zeros(shape=(3000,))
#     validation_label_preprocess = np.zeros(shape=(1000,))
#     test_label_preprocess = np.zeros(shape=(1000,))
    

     # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    data_allfeatures = np.array(np.vstack((train_data,validation_data,test_data)))
    num_feature = data_allfeatures.shape[1]
    # print("Initial Number of Features:")
    # print(num_feature)
    column_indexes = np.arange(num_feature)
    #column_index = np.arange(num_feature)
    same_val = np.all(data_allfeatures == data_allfeatures[0,:], axis = 0)
    #features_indices = np.all(data_allfeatures != data_allfeatures[0,:],axis = 0)
    column_id = column_indexes[same_val]
    #feature_indices = column_index[features_indices]
    # print(column_id)
    listtotal = np.arange(784)
    global indices
    indices = 0
    indices = np.setdiff1d(listtotal,column_id)
    # print(indices)# - column_id
    # print(feature_indices)
    data_redfeatures = np.delete(data_allfeatures,column_id,axis=1)
    num_features = data_redfeatures.shape[1]
    # print("Final Number of Features:")
    # print(num_features)
    
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

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
    #FeedForward Process Start
    
    tdata_bias = np.ones(training_data.shape[0],dtype = np.float64)
    z_j_bias = np.ones(training_data.shape[0],dtype = np.float64)
    
    training_data = np.column_stack([training_data,tdata_bias]) 
       
    
    
    a_j = np.dot(training_data,w1.transpose())
    z_j = np.column_stack([sigmoid(a_j),z_j_bias])
    o_l = sigmoid(np.dot(z_j,w2.transpose()))
    
    #FeedForward Process Ends

    #Process Training Labels
    
    y_l = np.zeros((len(training_data),10))
    for col in range (0,len(training_data)):
        y_l[col][int(training_label[col])]=1
         
    
      
    #Calculate Error    
    log_pos_ol = np.log(o_l)     
    
    log_neg_ol = np.log(1-o_l)
    param = np.multiply(y_l,log_pos_ol) + np.multiply((1-y_l),log_neg_ol)
   
    total = np.sum(param)
    
    err = (total/(len(training_data)))
    err = np.negative(err)

    #Gradiant Descent 
    
    d_l = o_l - y_l
    grad_w2 = np.dot(d_l.transpose(),z_j)
    product = ((np.dot(d_l,w2))*((1 - z_j)*z_j))
    grad_w1 = np.dot(product.transpose(),training_data)
    grad_w1 = grad_w1[0:n_hidden,:]
    
    
    #Process of Regularization
    
    w1_2 = np.sum(np.square(w1))
    w2_2 = np.sum(np.square(w2))  
    
    val = (lambdaval/(2*len(training_data)))*(w1_2+w2_2)
    obj_val = err + val
    
    grad_w1 = (grad_w1 + (lambdaval*w1))/len(training_data)
    grad_w2 = (grad_w2 + (lambdaval*w2))/len(training_data)



    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    #Your code here
    d_bias = np.ones((data.shape[0], 1),dtype = np.uint8)

    data = np.column_stack([data, d_bias])

    output = sigmoid(np.dot(data, w1.transpose()))      #Create output layer
    
    o_bias = np.ones((output.shape[0], 1),dtype = np.uint8)   #Make Bias of size of output

    output = np.column_stack([output, o_bias])      #Add bias to output 

    labels = sigmoid(np.dot(output, w2.transpose()))    #Dot product gives the output 
    
    labels = np.argmax(labels, axis=1)  #Our output is the one with the max value
    return labels


def run(lm=0, no_of_hidden_layers=50):
    """**************Neural Network Script Starts here********************************"""
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = no_of_hidden_layers

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = lm

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)
    train_accuracy = np.mean((predicted_label.astype(float) == train_label.astype(float)))

    # find the accuracy on Training Dataset

    print('\n Training set Accuracy:' + str(100 * train_accuracy) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)
    validation_accuracy = np.mean((predicted_label == validation_label).astype(float))

    # find the accuracy on Validation Dataset

    print('\n Validation set Accuracy:' + str(100 * validation_accuracy) + '%')

    predicted_label = nnPredict(w1, w2, test_data)
    test_accuracy = np.mean((predicted_label == test_label).astype(float))

    # find the accuracy on Validation Dataset

    print('\n Test set Accuracy:' + str(100 * test_accuracy) + '%')

    return train_accuracy, validation_accuracy, test_accuracy

    # global indices
    # obj = [indices, n_hidden, w1, w2, lambdaval]
    # # selected_features is a list of feature indices that you use after removing unwanted features in feature selection step

    # pickle.dump(obj, open('params.pickle', 'wb'))

    # pic_load = pickle.load( open( "params.pickle", "rb" ) )

if __name__ == "__main__":
  run()