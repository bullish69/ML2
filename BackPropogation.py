from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
Y = data.target

X

Y

y = pd.get_dummies(Y).values

y

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=41)


def sigmoid(y_in):
    y_hat = 1/(1 + np.exp(-y_in))
    return y_hat


def error(y, y_hat):
    err = np.array((y-y_hat)**2).mean()
    return err


def accuracy(y, y_hat):
    acc = 0
    acc += np.argmax(y, axis=1) == np.argmax(y_hat, axis=1)
    return acc.mean()


def forward_pass(x_t, y_t, v, w):
    z_in = np.dot(x_t, v)
    z = []
    for i in z_in:
        temp = []
        for j in i:
            temp.append(sigmoid(j))
        temp = np.array(temp)
        z.append(temp)
    z = np.array(z)
    y_in = np.dot(z, w)
    y_hat = []
    for i in y_in:
        temp = []
        for j in i:
            temp.append(sigmoid(j))
        temp = np.array(temp)
        y_hat.append(temp)
    y_hat = np.array(y_hat)

    acc = accuracy(y_t, y_hat)
    err = error(y_t, y_hat)
    return z, y_hat, acc, err


def backward_pass(x_t, y_t, z, v, w, y_hat, lr):
    dy = (y_t-y_hat)*y_hat*(1-y_hat)
    dw = np.dot(z.transpose(), dy)
    w += lr*dw
    dz = np.dot(dy, w.T)*z*(1-z)
    dv = np.dot(x_t.T, dz)
    v += lr*dv
    return v, w


best = [0, None, None]
v = np.random.randn(4, 2)
w = np.random.randn(2, 3)
b = 0
training_acc = []
training_err = []
for i in range(100):
    z, y_hat, acc, err = forward_pass(X_train, Y_train, v, w)
    training_acc.append(acc)
    training_err.append(err)
    if(acc > best[0]):
        best = [acc, w, v]
    v, w = backward_pass(X_train, Y_train, z, v, w, y_hat, 0.01)


epoch = [i for i in range(1, 101)]


plt.plot(epoch, training_acc)
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.show()


plt.plot(epoch, training_err)
plt.xlabel("Epoch")
plt.ylabel("Training Error")
plt.show()


best = [0, None, None]
v = np.random.randn(4, 2)
w = np.random.randn(2, 3)
b = 0
testing_acc = []
testing_err = []
for i in range(100):
    z, y_hat, acc, err = forward_pass(X_test, Y_test, v, w)
    testing_acc.append(acc)
    testing_err.append(err)
    if(acc > best[0]):
        best = [acc, w, v]
    v, w = backward_pass(X_test, Y_test, z, v, w, y_hat, 0.01)


plt.plot(epoch, testing_acc)
plt.xlabel("Epoch")
plt.ylabel("Testing Accuracy")
plt.show()


plt.plot(epoch, testing_err)
plt.xlabel("Epoch")
plt.ylabel("Testing Error")
plt.show()
