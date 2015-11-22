import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model.logistic as sk_logit

data= pd.read_csv("wdbc.data.txt", header=None, sep=r"\s+")

radius=np.array(data[2])

texture=np.array(data[3])

classification=np.array(data[1])

#a1=np.column_stack((classification,radius))
#a2=np.column_stack((classification,texture))

malignant = np.where(classification == "M" )
benign = np.where(classification=="B")
plt.scatter(radius[malignant],texture[malignant], marker='o', c='b')
plt.scatter(radius[benign],texture[benign], marker='x', c='r')
plt.xlabel('radius')
plt.ylabel('texture')
plt.legend(['MALIGNANT', 'BENIGN'])
#plt.show()


training_data= np.transpose(np.vstack((radius,texture)))


trained_bot=sk_logit._fit_liblinear(training_data,classification)