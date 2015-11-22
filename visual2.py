import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f=open("wdbc.data.txt")

data= pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")

radius=np.array(data[2])

area=np.array(data[4])

classification=np.array(data[1])

#a1=np.column_stack((classification,radius))
#a2=np.column_stack((classification,texture))

malignant = np.where(classification == "M" )
benign = np.where(classification=="B")
plt.scatter(radius[malignant],area[malignant], marker='o', c='b')
plt.scatter(radius[benign],area[benign], marker='x', c='r')
plt.xlabel('radius')
plt.ylabel('area')
plt.legend(['MALIGNANT', 'BENIGN'])
plt.show()

