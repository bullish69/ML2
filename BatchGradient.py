from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.5, 2.5])
Y = np.array([0.2, 0.9])


def sigmoid(x, w, b):
    y_in = np.dot(x, w) + b
    y_hat = 1/(1 + np.exp(-y_in))
    return y_hat


def error(y, y_hat):
    mse = np.array((y-y_hat)**2).mean()
    return mse


def accuracy(y, y_hat):
    acc = np.array((y_hat/y)*100)
    acc = normalize([acc])
    return acc.mean()*100


def batch_gd(X, Y, epochs):
    w = -2
    c = 1
    b = -2
    err_list_batch = []
    acc_list_batch = []
    W = []
    B = []
    for i in range(epochs):
        temp_w = 0
        temp_b = 0
        for x, y in zip(X, Y):
            y_hat = sigmoid(x, w, b)
            temp_w += c*(y_hat-y)*y_hat*(1-y_hat)*x
            temp_b += c*(y_hat-y)*y_hat*(1-y_hat)
        temp_w = temp_w/len(Y)
        temp_b = temp_b/len(Y)
        w -= temp_w
        b -= temp_b
        W.append(w)
        B.append(b)
        err_list_batch.append(error(Y, sigmoid(X, w, b)))
        acc_list_batch.append(accuracy(Y, sigmoid(X, w, b)))
        print(f"After epoch {i+1}: Weight ==> {w} and Bias ==> {b}")
    return W, B, err_list_batch, acc_list_batch


wt2, bias2, err_batch, acc_batch = batch_gd(X, Y, 100)

plt.plot(wt2, err_batch)
plt.xlabel("Weight")
plt.ylabel("Error")
plt.show()

plt.plot(bias2, err_batch)
plt.xlabel("Bias")
plt.ylabel("Error")
plt.show()

plt.plot(epoch, err_batch)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()


plt.plot(epoch, acc_batch)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
