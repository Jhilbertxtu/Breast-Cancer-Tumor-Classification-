import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f=open("wdbc.data.txt")

data= pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")

smoothness=np.array(data[4])

texture=np.array(data[3])

classification=np.array(data[1])

#a1=np.column_stack((classification,radius))
#a2=np.column_stack((classification,texture))

malignant = np.where(classification == "M" )
benign = np.where(classification=="B")
plt.scatter(smoothness[malignant],texture[malignant], marker='o', c='b')
plt.scatter(smoothness[benign],texture[benign], marker='x', c='r')
plt.xlabel('smoothness')
plt.ylabel('texture')
plt.legend(['MALIGNANT', 'BENIGN'])
plt.show()