import time
import numpy as np
import json
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
from sklearn.metrics import confusion_matrix


def preprocess():
    """
     Input:
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
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    bias_term = np.full((n_data, 1), 1)

    # Adding bias term to train_data

    X = np.hstack((bias_term, train_data))

    W_X = np.matmul(X, initialWeights)
    reshaped_W_X = W_X.reshape(W_X.shape[0], 1)

    theta = sigmoid(reshaped_W_X)

    log_theta = np.log(theta)
    log_one_minus_theta = np.log(1.0 - theta)

    error_term = np.sum((labeli * log_theta) + ((1.0 - labeli) * log_one_minus_theta))
    error = (- 1.0 * error_term) / n_data

    error_grad_term = np.sum((theta - labeli) * X, axis=0)
    error_grad = error_grad_term / n_data

    # print(error)
    # print(error_grad)
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    bias_term = np.full((data.shape[0], 1), 1)
    X = np.hstack((bias_term, data))

    theta_predict = sigmoid(np.matmul(X, W))

    label_i = np.argmax(theta_predict, axis=1)

    label = label_i.reshape((data.shape[0], 1))

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    bias_term = np.full((n_data, 1), 1)

    # Adding bias term to train_data

    X = np.hstack((bias_term,train_data))

    reshape_W = params.reshape((n_feature+1,n_class))
    W_X = np.matmul(X,reshape_W)

    num = np.exp(W_X)
    den = np.sum(num, axis=1)
    reshaped_den = den.reshape(den.shape[0],1)

    theta = num / reshaped_den
    log_theta = np.log(theta)

    error = - np.sum(np.sum(Y * log_theta))

    error_grad = np.ndarray.flatten(np.matmul(np.transpose(X),(theta - labeli)))
    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    bias_term = np.full((data.shape[0], 1), 1)

    # Adding bias term to train_data

    X = np.hstack((bias_term,data))
    W_X = np.matmul(X, W)

    num = np.exp(W_X)
    den = np.sum(num, axis=1)
    reshaped_den = den.reshape(den.shape[0], 1)

    theta_predict = num / reshaped_den

    label = np.argmax(theta_predict, axis=1)
    label = label.reshape(data.shape[0], 1)

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# # Logistic Regression with Gradient Descent
# W = np.zeros((n_feature + 1, n_class))
# initialWeights = np.zeros((n_feature + 1, 1))
# opts = {'maxiter': 110}
# print("Train data")
# for i in range(n_class):
#     labeli = Y[:, i].reshape(n_train, 1)
#     args = (train_data, labeli)
#     print("Class" + str(i+1))
#     nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
#     print(nn_params.fun)
#     # print((nn_params))
#     W[:, i] = nn_params.x.reshape((n_feature + 1,))
#     # print((W[:,i]))

# # Find the accuracy on Training Dataset
# predicted_label = blrPredict(W, train_data)
# print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
# print(confusion_matrix(train_label, predicted_label, labels = [0,1,2,3,4,5,6,7,8,9]))
# binary_logistic_Train_Accuracy = 100 * np.mean((predicted_label == train_label).astype(float))

# # Find the accuracy on Validation Dataset
# predicted_label = blrPredict(W, validation_data)
# print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
# print(confusion_matrix(validation_label, predicted_label, labels = [0,1,2,3,4,5,6,7,8,9]))
# binary_logistic_Validation_Accuracy = 100 * np.mean((predicted_label == validation_label).astype(float))


# # Find the accuracy on Testing Dataset
# predicted_label = blrPredict(W, test_data)
# print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
# print(confusion_matrix(test_label, predicted_label, labels = [0,1,2,3,4,5,6,7,8,9]))
# binary_logistic_Test_Accuracy = 100 * np.mean((predicted_label == test_label).astype(float))



# print("Test data")
# # number of test samples
# n_test = test_data.shape[0]

# # number of features
# n_feature = test_data.shape[1]

# Y = np.zeros((n_test, n_class))
# for i in range(n_class):
#     Y[:, i] = (test_label == i).astype(int).ravel()

# W = np.zeros((n_feature + 1, n_class))
# initialWeights = np.zeros((n_feature + 1, 1))
# opts = {'maxiter': 110}
# for i in range(n_class):
#     labeli = Y[:, i].reshape(n_test, 1)
#     args = (test_data, labeli)
#     print("Class" + str(i+1))
#     nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
#     print(nn_params.fun)
#     W[:, i] = nn_params.x.reshape((n_feature + 1,))



# """
# Script for Extra Credit Part
# """
# # FOR EXTRA CREDIT ONLY
# W_b = np.zeros((n_feature + 1, n_class))
# initialWeights_b = np.zeros((n_feature + 1, n_class))
# opts_b = {'maxiter': 110}

# print("Train data")

# # number of training samples
# n_train = train_data.shape[0]

# # number of features
# n_feature = train_data.shape[1]

# Y = np.zeros((n_train, n_class))
# for i in range(n_class):
#     Y[:, i] = (train_label == i).astype(int).ravel()

# args_b = (train_data, Y)
# nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
# print("Multi class")
# print(nn_params.fun)
# # print((nn_params))
# W_b = nn_params.x.reshape((n_feature + 1, n_class))
# # print((W_b))

# # Find the accuracy on Training Dataset
# predicted_label_b = mlrPredict(W_b, train_data)
# print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
# print(confusion_matrix(train_label, predicted_label_b, labels = [0,1,2,3,4,5,6,7,8,9]))
# multi_logistic_Train_Accuracy = 100 * np.mean((predicted_label_b == train_label).astype(float))

# # Find the accuracy on Validation Dataset
# predicted_label_b = mlrPredict(W_b, validation_data)
# print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')
# print(confusion_matrix(validation_label, predicted_label_b, labels = [0,1,2,3,4,5,6,7,8,9]))
# multi_logistic_Validation_Accuracy = 100 * np.mean((predicted_label_b == validation_label).astype(float))

# # Find the accuracy on Testing Dataset
# predicted_label_b = mlrPredict(W_b, test_data)
# print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
# print(confusion_matrix(test_label, predicted_label_b, labels = [0,1,2,3,4,5,6,7,8,9]))
# multi_logistic_Test_Accuracy = 100 * np.mean((predicted_label_b == test_label).astype(float))

# print("Test data")

# # number of test samples
# n_test = test_data.shape[0]

# # number of features
# n_feature = test_data.shape[1]

# Y = np.zeros((n_test, n_class))
# for i in range(n_class):
#     Y[:, i] = (test_label == i).astype(int).ravel()

# W_b = np.zeros((n_feature + 1, n_class))
# initialWeights_b = np.zeros((n_feature + 1, n_class))
# opts_b = {'maxiter': 110}

# args_b = (test_data, Y)
# nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
# print(nn_params.fun)


# # set width of bar
# barWidth = 0.2
# fig = plt.subplots(figsize =(12, 8))
 
# # set height of bar
# x1 = [15, 30]
# y1 = [binary_logistic_Train_Accuracy, multi_logistic_Train_Accuracy]
# y2 = [binary_logistic_Validation_Accuracy, multi_logistic_Validation_Accuracy]
# y3 = [binary_logistic_Validation_Accuracy, multi_logistic_Test_Accuracy]

 
# # Set position of bar on X axis
# br1 = np.arange(len(y1))
# br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# # Make the plot
# plt.bar(br1, y3, color ='g', width = barWidth,
#         edgecolor ='grey', label ='Test Accuracy')
# plt.bar(br2, y2, color ='orange', width = barWidth,
#         edgecolor ='grey', label ='Validation Accuracy')
# plt.bar(br3, y1, color ='b', width = barWidth,
#         edgecolor ='grey', label ='Train Accuracy')



# # function to add value labels
# def addlabels(x,y, color):
#     for i in range(len(x)):
#         plt.text(i,y[i],y[i], ha = 'center', bbox=dict(facecolor=color, edgecolor='none'))

# addlabels(x1, y3, 'green')
# plt.figure(0)
# plt.legend()
# plt.xticks([r + barWidth for r in range(len(y1))], ["Binary", "Multi Class"])

# ax = plt.gca()
# ax.set_ylim([85, 95])

# # Adding Xticks
# plt.title('Logistic Regression')
# plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15)
# plt.grid(True)
# # plt.show()




# """
# Script for Support Vector Machine
# """

# print('\n\n--------------SVM-------------------\n\n')
# ##################
# # YOUR CODE HERE #
# ##################

# idx = np.random.choice(50000, 20000, replace = False)
# svm_data = train_data[idx,:]
# svm_label = train_label[idx,:]

# start_time=time.time()  
# linear_kernel_svm = svm.SVC(kernel='linear')
# linear_kernel_svm.fit(svm_data, svm_label)
# end_time=time.time()

# print('\n ----------SVM linear kernel------------')
# print('\n Training Accuracy :' + str(100 * linear_kernel_svm.score(train_data, train_label)) + '%')
# print('\n Validation Accuracy :' + str(100 * linear_kernel_svm.score(validation_data, validation_label)) + '%')
# print('\n Testing Accuracy :' + str(100 * linear_kernel_svm.score(test_data, test_label)) + '%')
# print("Train Time : "+str(end_time-start_time))

# # Radial basis function with value of gamma setting to 1
# start_time=time.time()  
# rbf_kernel_svm = svm.SVC(kernel='rbf', gamma = 1.0)
# rbf_kernel_svm.fit(svm_data, svm_label)
# end_time=time.time()

# print('\n ------------SVM rbf kernel with gamma = 1----------------')
# print('\n Training Accuracy :' + str(100 * rbf_kernel_svm.score(svm_data, svm_label)) + '%')
# print('\n Validation Accuracy :' + str(100 * rbf_kernel_svm.score(validation_data, validation_label)) + '%')
# print('\n Testing Accuracy :' + str(100 * rbf_kernel_svm.score(test_data, test_label)) + '%')
# print("Train Time : "+str(end_time-start_time))

# # Radial basis function with value of gamma setting to default
# start_time=time.time()  
# rbf_dfltG_svm = svm.SVC(kernel='rbf', gamma = 'auto')
# rbf_dfltG_svm.fit(svm_data, svm_label)
# end_time=time.time()

# print('\n ------------SVM rbf kernel with gamma = default----------------')
# print('\n Training Accuracy :' + str(100 * rbf_dfltG_svm.score(train_data, train_label)) + '%')
# print('\n Validation Accuracy :' + str(100 * rbf_dfltG_svm.score(validation_data, validation_label)) + '%')
# print('\n Testing Accuracy :' + str(100 * rbf_dfltG_svm.score(test_data, test_label)) + '%')
# print("Train Time : "+str(end_time-start_time))
# # Radial basis function with value of gamma setting to default and varying value of C
# accuracy = np.zeros((11,3), float)
# c_vals = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# inp = 0
# plot_data = []

# #iterating through C values
# for c in c_vals:
#     start_time=time.time()  
#     rbf_c_svm = svm.SVC(kernel = 'rbf', C = c, gamma = 'auto')
#     rbf_c_svm.fit(svm_data, svm_label.flatten())
#     end_time=time.time()
#     accuracy[inp][0] = 100 * rbf_c_svm.score(train_data, train_label)
#     accuracy[inp][1] = 100 * rbf_c_svm.score(validation_data, validation_label)
#     accuracy[inp][2] = 100 * rbf_c_svm.score(test_data, test_label)

#     print('\n -----------SVM rbf kernel and C:'+str(c)+'---------------')
#     print('\n Training Accuracy :' + str(accuracy[inp][0]) + '%')
#     print('\n Validation Accuracy :' + str(accuracy[inp][1]) + '%')
#     print('\n Testing Accuracy :' + str(accuracy[inp][2]) + '%')
#     print("Train Time : "+str(end_time-start_time))
#     plot_data.append([c, accuracy[inp][0] , accuracy[inp][1] , accuracy[inp][2], (end_time-start_time)])
#     inp += 1


# # import json
# # with open('plot_values.json', 'w') as out:
# #     out.write(json.dumps(plot_data))

# x1 = [x[0] for x in plot_data]
# y1 = [x[1] for x in plot_data]
# y2 = [x[2] for x in plot_data]
# y3 = [x[3] for x in plot_data]
# y4 = [x[4] for x in plot_data]


# plt.figure(1)
# plt.plot(x1, y1, color='blue', label='Training Accuracy', linewidth=1)
# plt.plot(x1, y2, color='orange', label='Validation Accuracy', linewidth=1)
# plt.plot(x1, y3, color='green', label='Test Accuracy', linewidth=1)
# plt.title('SVM Guassian Kernel Accurcy vs C', fontsize=18)
# plt.xlabel('C', fontsize=12)
# plt.ylabel('Accuracy', fontsize=12)
# ax = plt.gca()
# ax.set_ylim([92, 98])
# plt.legend()
# plt.grid(True)
# # plt.show()

# running on complete set
start_time=time.time()  
rbf_full_optimum = svm.SVC(kernel = 'rbf', gamma = 'auto', C = 100)
rbf_full_optimum.fit(train_data, train_label.flatten())
end_time=time.time()  

print('\n -----------RBF with FULL training set with best C:'+str(100)+'---------------')
print('\n Training Accuracy:' + str(100 * rbf_full_optimum.score(train_data, train_label)) + '%')
print('\n Validation Accuracy:' + str(100 * rbf_full_optimum.score(validation_data, validation_label)) + '%')
print('\n Testing Accuracy:' + str(100 * rbf_full_optimum.score(test_data, test_label)) + '%')
print("Train Time : "+str(end_time-start_time))
  



