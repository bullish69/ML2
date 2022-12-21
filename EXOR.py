import numpy as np


def step(Y_in):
    if Y_in < 0:
        return 0
    else:
        return 1


def perceptron(x, w, b):
    y_in = np.dot(x, w) + b
    return step(y_in)


def EXOR(x):
    xor1 = NAND(x)
    xor2 = OR(x)
    return AND(np.array([xor1, xor2]))


test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for i in test:
    print(i[0], "EXOR", i[1], 'is', EXOR(i))
