import csv
import json
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import random

# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i], ha = 'center', bbox=dict(facecolor='green', edgecolor='none'))


# # creating the dataset
data = {'0':7.4, '1':93.8 , '9':98.7,
        '999':99.1, '9999': 99.2}
courses = list(data.keys())
values = list(data.values())
x1=[0,1,9,999,9999]
x1=[0,1,2,3,4]
y1=[7.4, 93.8, 98.7, 99.1, 99.2]
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='green',
        width = 0.5)
addlabels(x1, y1)
plt.xlabel("No of Iterations", fontweight ='bold', fontsize = 15)
plt.ylabel("Test Accuracy", fontweight ='bold', fontsize = 15)
plt.title("Convolution Neural Networks")
# plt.xticks([r + 0 for r in range(len(y1))], x1)
plt.savefig("CNNAccuracyMInist" + ".jpg")



plt.figure()
barWidth = 0

y1=[2.308481 ,0.222608, 0.036481,  0.043888 , 0.04]
data = {'0':y1[0], '1':y1[1] , '9':y1[2],
        '999':y1[3], '9999': y1[4]}
courses = list(data.keys())
values = list(data.values())
plt.bar(courses, values, color='r', width = 0.5,
        edgecolor ='grey', label ='Loss')

plt.title('Convolution Neural Networks')
plt.xlabel('No of Iterations', fontweight ='bold', fontsize = 15)
plt.ylabel('Loss', fontweight ='bold', fontsize = 15)
# addlabels(x1, y2)
plt.savefig("CNNLossMInist.jpg")



plt.figure()
# data = {'1':31.32, '2':125.45978927612305, '3':150.13367414474487,
#         '5':158.2198293209076, '7': 158.79158473014832}
# courses = list(data.keys())
# values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
x1=[0,1,9,999,9999]
y1=[0, 24, 195, 2368, 28080]
data = {'0':y1[0], '1':y1[1] , '9':y1[2],
        '999':y1[3], '9999': y1[4]}
courses = list(data.keys())
values = list(data.values())
fig = plt.figure(figsize = (10, 5))
addlabels(x1, y1)
# creating the bar plot
plt.bar(courses, values, color ='orange',
        width = 0.5)
plt.xlabel("No of Iterations")
plt.ylabel("Training Time (in sec)")
plt.title('Convolution Neural Networks')
plt.savefig("CNNTraningTimeMInist" + ".jpg")




# # creating the dataset
data = {'0':10.1, '1':95.8 , '9':96.2,
        '999':96.5}
courses = list(data.keys())
values = list(data.values())
x1=[0,1,9,999]
x1=[0,1,2,3]
y1=[10.1, 95.8 , 96.2, 96.5]
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='green',
        width = 0.5)
addlabels(x1, y1)
plt.xlabel("No of Iterations", fontweight ='bold', fontsize = 15)
plt.ylabel("Test Accuracy", fontweight ='bold', fontsize = 15)
plt.title("Convolution Neural Networks")
# plt.xticks([r + 0 for r in range(len(y1))], x1)
plt.savefig("CNNAccuracyCelebA" + ".jpg")



plt.figure()
barWidth = 0

y1=[0.699685  ,0.146522 ,  0.124888 ,   0.11 ]
data = {'0':y1[0], '1':y1[1] , '9':y1[2],
        '999':y1[3]}
courses = list(data.keys())
values = list(data.values())
plt.bar(courses, values, color='r', width = 0.5,
        edgecolor ='grey', label ='Loss')

plt.title('Convolution Neural Networks')
plt.xlabel('No of Iterations', fontweight ='bold', fontsize = 15)
plt.ylabel('Loss', fontweight ='bold', fontsize = 15)
# addlabels(x1, y2)
plt.savefig("CNNLossMInistCelebA.jpg")



plt.figure()
# data = {'1':31.32, '2':125.45978927612305, '3':150.13367414474487,
#         '5':158.2198293209076, '7': 158.79158473014832}
# courses = list(data.keys())
# values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
x1=[0,1,9,999]
y1=[0, 1445, 24420, 48456]
data = {'0':y1[0], '1':y1[1] , '9':y1[2],
        '999':y1[3]}
courses = list(data.keys())
values = list(data.values())
fig = plt.figure(figsize = (10, 5))
addlabels(x1, y1)
# creating the bar plot
plt.bar(courses, values, color ='orange',
        width = 0.5)
plt.xlabel("No of Iterations")
plt.ylabel("Training Time (in sec)")
plt.title('Convolution Neural Networks')
plt.savefig("CNNTraningTimeCelebA" + ".jpg")




cm = [[2514 ,  4], [ 101  , 23]]

# Print the confusion matrix as text.
print(cm)

# Plot the confusion matrix as an image.
plt.matshow(cm)

# Make various adjustments to the plot.
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, range(2))
plt.yticks(tick_marks, range(2))
plt.xlabel('Predicted')
plt.ylabel('True')

# Ensure the plot is shown correctly with multiple plots
# in a single Notebook cell.
# plt.show()
plt.savefig("confusion_matrix_celebA.jpg")




