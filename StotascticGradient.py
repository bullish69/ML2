import numpy as np

X = np.array([0.5, 2.5]) 
Y = np.array([0.2, 0.9]) 

def sigmoid(x, w, b): 
  y_in = np.dot(x, w) + b 
  y_hat = 1/(1 + np.exp(-y_in)) 
  return y_hat 

def error(y, y_hat):
  mse = np.array((y-y_hat)**2).mean() 
  return mse 

from sklearn.preprocessing import normalize 
def accuracy(y, y_hat): 
  acc = np.array((y_hat/y)*100) 
  acc = normalize([acc]) 
  return acc.mean()*100 

def stochastic_gd(X, Y, epochs):
  w = -2 
  c = 1 
  b = -2 
  err_list = [] 
  acc_list = [] 
  W = [] 
  B = [] 
  for i in range(epochs): 
    for x, y in zip(X, Y):
      y_hat = sigmoid(x, w, b) 
      w -= c*(y_hat-y)*y_hat*(1-y_hat)*x 
      b -= c*(y_hat-y)*y_hat*(1-y_hat) 
    W.append(w) 
    B.append(b) 
    err_list.append(error(Y, sigmoid(X, w, b)))
    acc_list.append(accuracy(Y, sigmoid(X, w, b)))
    print(f"After epoch {i+1}: Weight ==> {w} and Bias ==> {b}") 
  return W, B, err_list, acc_list

wt, bias, err, acc = stochastic_gd(X, Y, 100)

epoch = [i for i in range(1, 101)]

import matplotlib.pyplot as plt 
plt.plot(wt, err)
plt.xlabel("Weight") 
plt.ylabel("Error")
plt.show()

import matplotlib.pyplot as plt 
plt.plot(bias, err)
plt.xlabel("Bias") 
plt.ylabel("Error")
plt.show()

import matplotlib.pyplot as plt 
plt.plot(epoch, err)
plt.xlabel("Epoch") 
plt.ylabel("Error")
plt.show()


import matplotlib.pyplot as plt 
plt.plot(epoch, acc)
plt.xlabel("Epoch") 
plt.ylabel("Accuracy")
plt.show()