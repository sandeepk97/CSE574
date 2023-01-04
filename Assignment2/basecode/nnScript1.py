import json
import pickle
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import time
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


def run(lm=0, no_of_hidden_layers=50, iter=50):
    """**************Neural Network Script Starts here********************************"""
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = no_of_hidden_layers

    # set the number of nodes in output unit
    n_class = 10

    start_time = time.time()
    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = lm

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': iter}  # Preferred value.

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

    end_time = time.time()

    total_time = end_time - start_time

    global indices
    obj = [indices, n_hidden, w1, w2, lambdaval]
    # selected_features is a list of feature indices that you use after removing unwanted features in feature selection step

    pickle.dump(obj, open('params.pickle', 'wb'))

    pic_load = pickle.load( open( "params.pickle", "rb" ) )
    print(pic_load)

    return train_accuracy, validation_accuracy, test_accuracy, total_time



if __name__ == "__main__":
  run(0, 28,150)

# params.pickle for optimal lambda
# [array([ 12,  13,  14,  15,  32,  33,  34,  35,  36,  37,  38,  39,  40,
#         41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  58,  59,
#         60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
#         73,  74,  75,  76,  77,  78,  79,  80,  81,  86,  87,  88,  89,
#         90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102,
#        103, 104, 105, 106, 107, 108, 109, 110, 113, 114, 115, 116, 117,
#        118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
#        131, 132, 133, 134, 135, 136, 137, 138, 139, 141, 142, 143, 144,
#        145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
#        158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 169, 170, 171,
#        172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184,
#        185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
#        198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
#        211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
#        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236,
#        237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249,
#        250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262,
#        263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275,
#        276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288,
#        289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301,
#        302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314,
#        315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327,
#        328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
#        341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353,
#        354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366,
#        367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379,
#        380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392,
#        393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405,
#        406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418,
#        419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431,
#        432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444,
#        445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457,
#        458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470,
#        471, 472, 473, 474, 475, 477, 478, 479, 480, 481, 482, 483, 484,
#        485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497,
#        498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510,
#        511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523,
#        524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536,
#        537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549,
#        550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 561, 562, 563,
#        564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576,
#        577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589,
#        590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602,
#        603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615,
#        616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628,
#        629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641,
#        642, 643, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655,
#        656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668,
#        669, 670, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684,
#        685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697,
#        698, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713,
#        714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726,
#        731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743,
#        744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 760, 761, 762,
#        763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775,
#        776, 777, 778, 779]), 28, array([[ 8.87985265e-03, -4.09231790e-03,  3.54822318e-02, ...,
#          7.50403350e-02, -6.16079677e-02, -7.07195189e-01],
#        [ 2.41373649e-02, -2.49162240e-02,  1.66866661e-02, ...,
#         -7.72195733e-02,  7.17057831e-02, -4.78483365e-02],
#        [ 5.25537725e-02,  5.99277655e-02, -2.28825118e-03, ...,
#          5.45079903e-02, -2.05377780e-02,  4.07264135e-01],
#        ...,
#        [-5.03556254e-02, -1.59654524e-02, -4.86360411e-02, ...,
#          4.70981647e-02, -6.75068646e-02, -7.71086316e-01],
#        [ 1.79590232e-02, -2.61939309e-02, -7.66111195e-02, ...,
#         -4.68548873e-02,  5.90797564e-02, -4.91367472e-01],
#        [ 6.74083766e-04,  3.34989679e-02,  1.24815607e-02, ...,
#          2.77152010e-02,  2.15480134e-02, -7.10377868e-01]]), array([[-0.27005762,  1.31932236, -1.07436992, -1.1572096 , -0.58155036,
#         -1.24117642,  1.45848523, -4.955226  , -3.34494362,  1.48050505,
#          1.5339019 , -0.60235447, -1.88377769, -1.89591671, -1.98603556,
#         -3.60135242,  1.67927359, -3.05164763, -0.93815225,  1.45560027,
#          1.32189188, -0.64636916, -0.10429409,  0.24951016, -2.64178621,
#          2.55903219, -0.93965508, -1.20485621, -0.88005359],
#        [ 1.0213043 , -2.56607777, -1.43625838,  1.16962216, -1.56831882,
#          1.91245813, -0.47716456, -2.03086799,  0.88959886, -2.8028759 ,
#          0.75290013,  1.49027642, -1.84581457, -3.54790162,  1.94374801,
#         -0.65607279, -4.05653051,  0.54882688, -2.42046326, -1.15299566,
#         -0.03819214,  3.15127356,  0.28566678,  1.6372565 ,  3.04794884,
#         -1.63933129,  4.66054868,  1.06215126, -1.11866064],
#        [-5.96172878,  0.91732506, -1.6421917 , -1.61688872,  4.16211474,
#          0.5478047 , -0.08097341,  1.10663148,  3.5326375 ,  2.19935286,
#         -2.2058956 , -2.87695256, -0.19760719, -1.35934394, -0.96584145,
#         -3.68816088, -3.59195721, -2.45478958,  1.9226331 , -0.06869896,
#         -1.69040716, -1.08422916,  3.47941821,  0.92695687, -0.59074854,
#         -0.97953958, -0.49124639,  1.77864912, -1.73262793],
#        [ 0.82565641, -3.91168773, -1.64996597,  1.05590216,  0.43320973,
#         -0.63599064,  2.91684617,  1.47434704,  1.05051227,  0.19285923,
#         -0.38459955, -3.59897534, -4.51575474,  1.14018102, -1.80034614,
#          3.61718447,  2.79081619, -0.53583974, -0.46373592, -2.16654966,
#         -4.12081194, -2.49964485, -3.8704744 , -0.99296436,  1.71965014,
#         -1.82228386, -1.90796622,  0.23217086, -1.9705794 ],
#        [-0.00931438, -0.83384931, -1.71917527,  0.09565808, -1.72433558,
#          3.56568717, -3.47870928,  1.92683107,  1.2953256 , -3.43736962,
#          2.53758254,  0.84769532, -1.38221047,  0.93609512, -3.30816023,
#         -1.51320457, -0.36034215,  0.61455401,  1.40333421,  0.05915612,
#         -1.56261351,  0.90330062,  1.9679666 , -4.64621322, -3.46454597,
#         -0.03492782,  0.29863136,  3.21257938, -2.23679412],
#        [-0.11126886,  0.79238752, -0.69992851, -3.4555007 ,  2.95013326,
#         -0.46434777, -1.59303766,  1.74637236, -0.84365335,  0.89490846,
#          1.85119734,  3.3239118 ,  3.5833719 , -1.18263225, -1.28333056,
#          3.09848725, -3.70908153,  1.09372191, -4.04597253, -2.71201448,
#          5.5405535 , -3.76267999, -2.00563531, -2.06992574, -3.82205859,
#         -2.7749103 , -0.51429181,  0.09659134, -0.60307746],
#        [-0.8731525 , -0.10658959, -2.20091099,  0.75021943, -4.1323239 ,
#         -1.02287233,  1.29541369,  1.37433738, -3.93372667, -4.78823115,
#         -1.34313212,  1.29261569,  0.52281114, -2.1941814 ,  4.24759315,
#          2.19513468,  0.54228127, -3.76960999, -2.86544034,  0.64411589,
#         -0.4947708 , -1.25550641,  0.05269565, -2.9257729 ,  2.67324076,
#         -0.36015459, -1.06643635,  1.30985155, -2.38540563],
#        [-3.21698567, -1.06061717, -1.42683345,  4.63735373, -1.03852346,
#          2.45837369, -3.43371147, -3.32519899, -0.11858474,  2.18465795,
#         -0.93270464, -0.69584842, -2.55025919,  3.40478248,  1.98870066,
#          1.89743774, -2.27826837,  0.79062066, -0.32822172, -2.45994904,
#          0.73669313,  0.84036565, -2.70491651,  0.14622583, -2.16518035,
#          2.81060619, -1.06761963,  0.98262285, -2.22721974],
#        [ 1.01127636,  3.9000867 , -1.94647569, -2.70099757, -6.0096178 ,
#         -2.64836088, -1.03680417, -2.31695437, -0.70297942, -1.24806621,
#         -2.03509273, -1.15443222,  2.02549349, -2.87840149, -2.5089184 ,
#         -1.70283147,  1.1076686 ,  4.22629426,  3.71925895, -3.02489446,
#         -3.50784647, -4.4743519 ,  1.20704836, -1.07110718,  2.2823569 ,
#         -2.9206341 , -2.77246524, -3.42307769, -2.15037808],
#        [ 2.73425592, -1.91231538, -0.51004308, -3.13100495, -0.96114641,
#         -5.8206914 , -2.68802559,  2.10065632, -2.3624383 , -3.3518561 ,
#         -2.6251817 ,  1.27673032,  1.21396703,  1.33550174, -2.46830772,
#         -0.6488556 , -0.19645207, -2.38004558,  1.55291439,  3.45407719,
#          0.87695184,  1.32891472, -1.71539083,  3.469981  , -1.72079439,
#          0.43169262, -0.61464247, -4.58521706, -0.99014025]]), 0]

# # b = range(0,20,10)
# # k = [ 4, 8]
# b = range(0,70,5)
# k = [ 4, 8, 12, 16, 20, 24, 28]


# # combinations = [[0,4]]
# data = {}
# for iter in [50, 100, 110, 150, 200, 250, 400]:
#     combinations = [[f,s] for f in b for s in k]
#     for index, comb in enumerate(combinations):
#         print("Training for lambda: " + str(comb[0]) + ", no of hidden layers: " +  str(comb[1])  + ", no of iteration: " +  str(iter))
#         train_accuracy, validation_accuracy, test_accuracy, ttime = run(comb[0], comb[1], iter)
#         comb.append(train_accuracy)
#         comb.append(validation_accuracy)
#         comb.append(test_accuracy)
#         comb.append(ttime)
#     data[iter] = combinations
#     with open('auto_nnScript.json', 'w') as out_file:
#         out = json.dumps(data)
#         out_file.write(out)

# # arr = np.asarray(combinations)
# # pd.DataFrame(arr).to_csv('auto_nnScript.csv')  


# with open('auto_nnScript.json', 'r') as in_file:
#     combinations = json.load(in_file)['combinations']