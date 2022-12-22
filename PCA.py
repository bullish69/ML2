import keras 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from keras.models import Model 
from keras.layers import Dense, Input 
from sklearn import datasets 
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler


iris = datasets.load_iris() 
X = iris.data 
Y = iris.target 
target_names = iris.target_names


scaler = MinMaxScaler() 
scaler.fit(X) 
X_scaled = scaler.transform(X)


def plot3clusters(X, title, vtitle): 
  plt.figure() 
  colors = ['red', 'green', 'blue'] 
  lw = 2
  for color, i, target_name in zip(colors, [0, 1, 2], target_names): 
    plt.scatter(X[Y==i, 0], X[Y==i, 1], color=color, alpha=1., lw=lw, label=target_name) 
    plt.legend(loc='best', shadow=False, scatterpoints=1) 
    plt.title(title) 
    plt.xlabel(vtitle + "1") 
    plt.ylabel(vtitle + "2") 
    plt.show()

  
pca = decomposition.PCA() 
pca_transformed = pca.fit_transform(X_scaled) 
plot3clusters(pca_transformed[:, :2], 'PCA', 'PC')