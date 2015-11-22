__author__ = 'Prateek'


# Code source: Gl Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets,metrics,svm,tree

def LogReg(s,e):
    f=open("wdbc.data.txt")
    data= pandas.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y= np.where(Y=='M', 1, 0)
    X=X[:,s:e]
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)

    Z = logreg.predict(X)


    print(metrics.classification_report(Y, Z))
    print(metrics.confusion_matrix(Y,Z))

def SVM(s,e):
    f=open("wdbc.data.txt")
    data= pandas.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y= np.where(Y=='M', 1, 0)
    X=X[:,s:e]

    logreg=svm.SVC()
    logreg.fit(X, Y)

    Z = logreg.predict(X)
    print(metrics.classification_report(Y, Z))
    print(metrics.confusion_matrix(Y,Z))

def DTC(s,e):
    f=open("wdbc.data.txt")
    data= pandas.read_csv("wdbc.data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y= np.where(Y=='M', 1, 0)
    X=X[:,s:e]
    logreg = tree.DecisionTreeClassifier()
    logreg.fit(X, Y)

    Z = logreg.predict(X)

    print(metrics.classification_report(Y, Z))
    print(metrics.confusion_matrix(Y,Z))

'''
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
'''