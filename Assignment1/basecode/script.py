from cProfile import label
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    labels = np.unique(y)
    
    means_matrix = np.zeros((len(labels), X.shape[1]))
    for y1 in labels:
        index = np.where(y == y1)[0]
        x1 = X[index]     
        means_matrix[int(y1)-1] = x1.mean(axis=0)
    
    xTranspose = np.transpose(X)
    covmat = np.cov(xTranspose)
    return means_matrix, covmat

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    N = np.shape(Xtest)[0]
    m = np.shape(means)[0]
    labelsCount = np.shape(means)[1]
    ll = np.zeros((N, m))
    for i in range(N):
        g = 1 / np.sqrt((2*np.pi**labelsCount)*det(covmat))
        for h in range(m):
            b = Xtest[i, :] - means[int(h) - 1]
            t = (-1/2)*np.dot(np.dot(b.T, inv(covmat)), b)
            ll[i,int(h)-1] = g * np.e**t 
            
    ypred = [list(row).index(max(list(row)))+1 for row in ll]
    
    accuracy = 0
    for _, (ypre,ytes) in enumerate(zip(ypred, ytest)):
        if ypre == ytes:
            accuracy += 1
    return accuracy / len(ypred), np.array(ypred)

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    labels = np.unique(y)
    k = np.shape(labels)[0]
    d = np.shape(X)[1]
    means = np.zeros([k,d])
    covmats = []

    for i,y1 in enumerate(labels):
        x1 = X[np.where(y == y1)[0], ]   
        m = x1.mean(axis=0)
        means[i,] = m
        covmats.append(np.cov(np.transpose(x1)))
    return means, covmats


    
def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    N = np.shape(Xtest)[0]
    a = np.unique(ytest)
    ll = np.zeros((N, means.shape[0]))
    for i in range(Xtest.shape[0]):
        for h in range(means.shape[0]):
            index = int(h)-1
            b = Xtest[i, :] - means[index]
            t = (-1/2)*np.dot(np.dot(b.T, inv(covmats[index])), b)
            g = 1 / np.sqrt((2*np.pi**means.shape[1])*det(covmats[index]))
            ll[i,index] = g * np.e**t 
            
    ypred=[list(row).index(max(list(row)))+1 for row in ll]
    
    accuracy = 0
    for _, (ypre,ytes) in enumerate(zip(ypred, ytest)):
        if ypre == ytes:
            accuracy += 1
    return accuracy / len(ypred), np.array(ypred)

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    w = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,y))  
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:mse
    
    # IMPLEMENT THIS METHOD
    # return mse
    rmse = np.sqrt((1.0/Xtest.shape[0])*np.sum(np.square((ytest-np.dot(Xtest,w)))))
    return rmse

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    I = np.eye(X.shape[1],dtype=int) 
    w = np.dot(np.linalg.inv(np.dot(X.T,X) + np.dot(lambd,I)),np.dot(X.T,y))
    return w


def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    w = w.reshape(65,1)    
    error = 0.5*np.dot((y - np.dot(X,w)).transpose(),(y - np.dot(X,w))) + (0.5*lambd*np.dot(w.transpose(),w))
   
    error_grad = ((np.dot(np.dot(X.transpose(),X), w)) - (np.dot(X.transpose(),y))) + (lambd*w)
    error_grad = error_grad.flatten()
    
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    Xp = np.ones((x.shape[0], p + 1))
    n = 1
    while n < p+1:
      #changing nth column of the Xp into x power of n
      Xp[:, n] = x ** n
      n += 1
    return Xp       

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

print("X: ")
print(X[0:5])
print("y: ")
print(y[0:5])
# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
min_mse3 = 9999999.0
mses3 = np.zeros((k,1))
min_weights = np.empty([X.shape[1], 1])
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    if mses3[i] < min_mse3:
        min_mse3 = mses3[i]
        min_lambda = lambd
        min_weights = w_l
    i = i + 1
print('optimal value for λ: '+ str(min_lambda) + ' that gave the lowest MSE error of '+str(min_mse3))
print('sum (OLE Regression): ' + str(np.sum(w_i)) + ', Sum (Ridge Regression): '  + str( np.sum(min_weights)))
print('variance (OLE Regression): ' + str(np.var(w_i)) + ', Variance (Ridge Regression): ' + str( np.var(min_weights)))
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.xlabel('λ')
plt.ylabel('MSE')
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.xlabel('λ')
plt.ylabel('MSE')
plt.show()

plt.figure()
plt.title("Weight comparison")
plt.plot(range(0, w_i.shape[0]),w_i)
plt.plot(range(0, min_weights.shape[0]),min_weights)
plt.xlabel('Weight learnt')
plt.legend(('OLE','Ridge regression'))
plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

print(mses5)
df = pd.DataFrame()
df['p'] = range(pmax)
df['Test Error with λ=0'] = [x[0] for x in mses5]
df['Test Error with λ=0.06'] = [x[1] for x in mses5]
print(df)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.ylabel('MSE')
plt.xlabel('p')
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.ylabel('MSE')
plt.xlabel('p')
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()