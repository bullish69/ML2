import numpy as np


def step(Y_in):
    if Y_in < 0:
        return 0
    else:
        return 1


def perceptron(x, w, b):
    y_in = np.dot(x, w) + b
    return step(y_in)


def OR(x):
    w = np.array([1, 1])
    b = -1
    return perceptron(x, w, b)


test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for i in test:
    print(i[0], "OR", i[1], 'is', OR(i))
