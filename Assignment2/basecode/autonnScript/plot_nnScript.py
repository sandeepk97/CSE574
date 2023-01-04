import csv
import json
import random
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# from nnScript import run
# b = range(0,20,10)
# k = [ 4, 8]
# b = range(0,70,5)
# k = [ 4, 8, 12, 16, 20, 24, 28]

# combinations = [[f,s] for f in b for s in k]
# # combinations = [[0,4]]
# for index, comb in enumerate(combinations):
#     print("Training for lambda: " + str(comb[0]) + ", no of hidden layers: " +  str(comb[1]))
#     train_accuracy, validation_accuracy, test_accuracy = run(comb[0],comb[1])
#     comb.append(train_accuracy)
#     comb.append(validation_accuracy)
#     comb.append(test_accuracy)

# arr = np.asarray(combinations)
# pd.DataFrame(arr).to_csv('auto_nnScript.csv')  

# data = {}
# data['combinations'] = combinations
# with open('auto_nnScript.json', 'w') as out_file:
#     out = json.dumps(data)
#     out_file.write(out)

λ = 0
no_of_hidden = 4
iterations = 150 
i=16 # [50, 100, 110, 150, 200, 250, 400]

with open('auto_nnScript.json', 'r') as in_file:
    filteredComb = []
    combinations = json.load(in_file)[str(iterations)]
    for index, comb in enumerate(combinations):
        if comb[1] == no_of_hidden:
        # if comb[0] == λ:
            filteredComb.append(comb)
    
combinations = filteredComb


outfn = 'values.txt'
outf = open(outfn, 'w+')
wr = csv.writer(outf, delimiter=" ")

# fig = plt.figure()
# # ax = plt.axes(projection ='3d')    
# ax = fig.add_subplot(111, projection='3d')    
# # ax.scatter([x[0] for x in combinations], [x[1] for x in combinations], [x[2] for x in combinations])
# xh,yh,zh,maxi=0, 0, 0,0
# for xc, yc, zc in zip([x[0] for x in combinations], [x[1] for x in combinations], [x[i+4] for x in combinations]):
#     if zc > maxi:
#         wr.writerow([xc, yc, zc])
#         xh,yh,zh = xc, yc, zc
#         maxi = zc
# label = 'optimal(λ, m, test accuracy) = (' + str(xh) + ' ,' + str(yh) + ' ,' + str(zh) + ')'
# ax.text(xh,yh,zh, label)

# surf = ax.plot_trisurf([x[0] for x in combinations], [x[1] for x in combinations], [x[i+4] for x in combinations], cmap=cm.jet, linewidth=0.1)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_title('test accuracy for iterations: ' + str(iterations))
# ax.set_xlabel('λ')
# ax.set_ylabel('no of hidden nodes (m)')
# ax.set_zlabel('test accuracy')

# # plt.show()
# plt.savefig("3dplotiterations" + str(iterations) + ".jpg")



# Fixing random state for reproducibility
np.random.seed(0)




# set width of bar
barWidth = 0.3
fig = plt.subplots(figsize =(12, 8))
 
# set height of bar
x1 = [x[0] for x in combinations]
y1 = [x[2] for x in combinations]
y2 = [x[i+3] for x in combinations]
y3 = [x[i+4] for x in combinations]
y4 = [x[i+5] for x in combinations]

 
# Set position of bar on X axis
br1 = np.arange(len(y1))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, y3, color ='g', width = barWidth,
        edgecolor ='grey', label ='Test Accuracy')
plt.bar(br2, y2, color ='b', width = barWidth,
        edgecolor ='grey', label ='Validation Accuracy')
plt.bar(br3, y1, color ='y', width = barWidth,
        edgecolor ='grey', label ='Train Accuracy')



# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i], ha = 'center', bbox=dict(facecolor='green', edgecolor='none'))

addlabels(x1, y3)
# Adding Xticks
# plt.title('Accuracy for λ = ' + str(λ) + ', iterations = ' + str(iterations))
# plt.xlabel('no of hidden nodes (m)', fontweight ='bold', fontsize = 15)
# plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15)

plt.title('Accuracy for no of hidden nodes(m) = ' + str(no_of_hidden) + ', iterations = ' + str(iterations))
plt.xlabel('λ', fontweight ='bold', fontsize = 15)
plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15)

plt.xticks([r + barWidth for r in range(len(y1))], x1)
 
plt.legend()
# plt.show()

plt.savefig("LambdaValueVsAccuracyfor" + str(no_of_hidden) + "hiddenlayers" + ".jpg")
# plt.savefig("AccuracyVsHiddenfor" + "λ = " + str(λ) + "iterations" + str(iterations) + ".jpg")


# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# br1 = np.arange(len(y1))

# plt.bar(br1, y4, color=colors, width = 0.9,
#         edgecolor ='grey', label ='Training Time')

# plt.title('Training time for λ = ' + str(λ) + ', iterations = ' + str(iterations))
# plt.xlabel('no of hidden nodes (m)', fontweight ='bold', fontsize = 15)
# plt.ylabel('Training time (in sec)', fontweight ='bold', fontsize = 15)

# plt.xticks([r + barWidth for r in range(len(y1))], x1)

# plt.savefig("TrainingtimevsHiddennodesfor" + "λ = " + str(λ) + "iterations" + str(iterations) + ".jpg")



