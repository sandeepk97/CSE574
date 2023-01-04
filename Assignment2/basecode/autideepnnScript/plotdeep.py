import csv
import json
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import random

# b = range(0,20,10)
# k = [ 4, 8]
b = [0.0001, 0.001]
k = [0, 25, 50, 75, 100]

# combinations = [[f,s] for f in b for s in k]
# data = {}
# for index, comb in enumerate(combinations):
#     print("Training for lambda: " + str(comb[0]) + ", training epochs: " +  str(comb[1]))
#     acuracy, test_loss, ttime = run(comb[0], comb[1])
#     comb.append(acuracy)
#     comb.append(test_loss)
#     comb.append(ttime)
# data['combinations'] = combinations
# with open('auto_deepnnScript.json', 'w') as out_file:
#     out = json.dumps(data)
#     out_file.write(out)

# arr = np.asarray(combinations)
# pd.DataFrame(arr).to_csv('auto_deepnnScript.csv')  


learning_rate = 0.001
layer = 2

with open('output' + str(layer) + '.json', 'r') as in_file:
    filteredComb = []
    combinations = json.load(in_file)['combinations']
    for index, comb in enumerate(combinations):
        # if comb[1] == no_of_hidden:
        if comb[0] == learning_rate:
            filteredComb.append(comb)
    
combinations = filteredComb

outfn = 'values.txt'
outf = open(outfn, 'w+')
wr = csv.writer(outf, delimiter=" ")


# ax.plot_surface(x, y, z, cmap="autumn_r", lw=0, rstride=1, cstride=1)

# set width of bar
barWidth = 0
fig = plt.subplots(figsize =(12, 8))
i=0
# set height of bar
x1 = [x[1] for x in combinations]
y1 = [x[2] for x in combinations]
y2 = [x[i+3] for x in combinations]
y3 = [x[i+4] for x in combinations]

 
# Set position of bar on X axis
br1 = np.arange(len(x1))
# br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
# plt.bar(br1, y2, color ='g', width = barWidth,
#         edgecolor ='grey', label ='Test Accuracy')
# plt.bar(br2, y1, color ='b', width = barWidth,
#         edgecolor ='grey', label ='Train Accuracy')
# plt.bar(br3, y3, color ='y', width = barWidth,
#         edgecolor ='grey', label ='Training Time')



# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i], ha = 'center', bbox=dict(facecolor='green', edgecolor='none'))

# Adding Xticks
# plt.title('Accuracy for λ = ' + str(λ) + ', iterations = ' + str(iterations))
# plt.xlabel('no of hidden nodes (m)', fontweight ='bold', fontsize = 15)
# plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15)

# plt.title('Accuracy for no of hidden nodes(m) = ' + str(no_of_hidden) + ', iterations = ' + str(iterations))
# plt.xlabel('λ', fontweight ='bold', fontsize = 15)
# plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15)

# plt.title('Deep Neural Network for learning rate = ' + str(learning_rate))
# plt.xlabel('Training Epochs', fontweight ='bold', fontsize = 15)
# plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15)

# plt.title('Deep Neural Network for learning rate = ' + str(learning_rate))
# plt.xlabel('Training Epochs', fontweight ='bold', fontsize = 15)
# plt.ylabel('Training Time', fontweight ='bold', fontsize = 15)

# plt.xticks([r + barWidth for r in range(len(y2))], x1)
 
plt.legend()
# plt.show()


# plt.savefig("TrainingEpochsVsTime" + str(layer) + "for" + str(learning_rate) + ".jpg")
# plt.savefig("TrainingEpochsVsAccuracyLayer" + str(layer) + "for" + str(learning_rate) + ".jpg")
# plt.savefig("LambdaValueVsAccuracyfor" + str(no_of_hidden) + "hiddenlayers" + ".jpg")
# plt.savefig("AccuracyVsHiddenfor" + "λ = " + str(λ) + "iterations" + str(iterations) + ".jpg")


colors = random.shuffle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

# br1 = np.arange(len(y1))
# plt.bar(br1, y1, color=colors, width = 0.5,
#         edgecolor ='grey', label ='Test Accuracy')

# plt.title('Deep Neural Network for learning rate = ' + str(learning_rate))
# plt.xlabel('Training Epochs', fontweight ='bold', fontsize = 15)
# plt.ylabel('Test Accuracy', fontweight ='bold', fontsize = 15)
# addlabels(x1, y1)
# plt.xticks([r + barWidth for r in range(len(y1))], x1)
# plt.savefig("TrainingEpochsVsAccuracyLayer" + str(layer) + "for" + str(learning_rate) + ".jpg")




# plt.figure()
# br2 = np.arange(len(y1))
# plt.bar(br1, y2, color='r', width = 0.5,
#         edgecolor ='grey', label ='Loss')

# plt.title('Deep Neural Network for learning rate = ' + str(learning_rate))
# plt.xlabel('Training Epochs', fontweight ='bold', fontsize = 15)
# plt.ylabel('Loss', fontweight ='bold', fontsize = 15)
# # addlabels(x1, y2)
# plt.xticks([r + barWidth for r in range(len(y1))], x1)
# plt.savefig("TrainingEpochsVsLossLayer" + str(layer) + "for" + str(learning_rate) + ".jpg")


# plt.figure()
# br2 = np.arange(len(y1))
# plt.bar(br1, y3, color='b', width = 0.5,
#         edgecolor ='grey', label ='Training time')

# plt.title('Deep Neural Network for learning rate = ' + str(learning_rate))
# plt.xlabel('Training Epochs', fontweight ='bold', fontsize = 15)
# plt.ylabel('Training time', fontweight ='bold', fontsize = 15)
# # addlabels(x1, y3)
# plt.xticks([r + barWidth for r in range(len(y1))], x1)
# plt.savefig("TrainingEpochsVsTimeLayer" + str(layer) + "for" + str(learning_rate) + ".jpg")


# # creating the dataset
data = {'1':96.24, '2':87.73, '3':88.19,
        '5':78.91, '7': 77.28}
courses = list(data.keys())
values = list(data.values())
x1=[1,2,3,5,7]
y1=values
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='green',
        width = 0.4)
addlabels(x1, y1)
plt.xlabel("No of Hidden Layers")
plt.ylabel("Test Accuracy")
plt.title("Comparision of Simple and Deep Neural Networks")
# plt.xticks([r + barWidth for r in range(len(y1))], x1)
plt.savefig("ComparisionwithhiddenLayersAccuracy" + ".jpg")



plt.figure()
data = {'1':691.235711812973, '2':125.45978927612305, '3':150.13367414474487,
        '5':158.2198293209076, '7': 158.79158473014832}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='orange',
        width = 0.4)
 
plt.xlabel("No of Hidden Layers")
plt.ylabel("Training Time (in sec)")
plt.title("Comparision of Simple and Deep Neural Networks")
plt.savefig("ComparisionwithhiddenLayersTraningTime" + ".jpg")

