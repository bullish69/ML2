import numpy as np


def step(Y_in):
    if Y_in < 0:
        return 0
    else:
        return 1


def perceptron(x, w, b):
    y_in = np.dot(x, w) + b
    return step(y_in)


def NOT(x):
    w = -1
    b = 0
    return perceptron(x, w, b)


test = np.array([0, 1])
for i in test:
    print("NOT of", i, 'is', NOT(i))
