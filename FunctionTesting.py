import numpy as np
import math


listAll = []
a = [0.9, 0.3, 202, 22, 19, 7.4, 4.1, 1.2]
b = [0.7, 0.1, 187, 16, 18, 6.8, 3.3, 0.9]
c = [10.9, 5.5, 699, 64, 100, 7.5, 3.2, 0.74]
d = [7.3, 4.1, 490, 60, 68, 7, 3.3, 0.89]
e = [0.9, 0.3, 310, 61, 58, 7, 3.4, 0.9]
# f = [1.3, 0.3, 201, 55, 67, 5, 3.2, 0.2]

sentiment = [-1, 1, 1, 1, -1]

listTemp = []

listAll.append(a)
listAll.append(b)
listAll.append(c)
listAll.append(d)
listAll.append(e)
# listAll.append(f)

listAll = np.array(listAll)
listATranspose = listAll.transpose()

# rbfMatrix = listAll.dot(listATranspose)

for i, var in enumerate(listATranspose):
    listTemp.append(var)

print(listATranspose)
print(listTemp[0])
