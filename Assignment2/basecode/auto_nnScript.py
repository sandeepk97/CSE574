import csv
import json
import numpy as np
import pandas as pd
from matplotlib import cm
import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from nnScript import run
# b = range(0,60,1)
# k = [ 4, 8, 12, 16, 20, 24, 28]
# b = range(0,20,10)
# k = [4, 8]
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

with open('auto_nnScript.json', 'r') as in_file:
    combinations = json.load(in_file)['combinations']


outfn = 'values.txt'
outf = open(outfn, 'w+')
wr = csv.writer(outf, delimiter=" ")
fig = plt.figure()
# ax = plt.axes(projection ='3d')    
ax = fig.add_subplot(111, projection='3d')    
# ax.scatter([x[0] for x in combinations], [x[1] for x in combinations], [x[2] for x in combinations])
xh,yh,zh,maxi=0, 0, 0,0
for xc, yc, zc in zip([x[0] for x in combinations], [x[1] for x in combinations], [x[4] for x in combinations]):
    if zc > maxi:
        wr.writerow([xc, yc, zc])
        xh,yh,zh = xc, yc, zc
        maxi = zc
label = '(' + str(xh) + ' ,' + str(yh) + ' ,' + str(zh) + ')'
ax.text(xh,yh,zh, label)

# x = np.reshape([x[0] for x in combinations], (10,10)).T
# y = np.reshape([x[1] for x in combinations], (10,10)).T
# z = np.reshape([x[2] for x in combinations], (10,10)).T
# ax.plot3D([x,y,z, 'green')
# ax.plot_wireframe(x,y,z , rstride=2, cstride=2)


# ax.plot_surface(x, y, z, cmap="autumn_r", lw=0, rstride=1, cstride=1)



# cs = ax.contour(x,y,z, 10, lw=3, colors="k", linestyles="solid")
# plt.clabel(cs, inline=1, fontsize=10)

surf = ax.plot_trisurf([x[0] for x in combinations], [x[1] for x in combinations], [x[4] for x in combinations], cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title('')
ax.set_xlabel('lambda')
ax.set_ylabel('hidden nodes')
ax.set_zlabel('test accuracy')

# Save the figure object as binary file
file = open(r"fig.pickle", "wb")
pickle.dump(fig, file)
file.close()
plt.show()


plot = pickle.load(open(r"fig.pickle", "rb"))
plot.show()



