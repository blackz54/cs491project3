import numpy as np


def perceptron_train(X, Y):
    Y = Y.flatten()
    Y = transform_labels(Y)
    w = [0] * len(X[0])
    b = 0
    count = 0
    epoch = 0

    while True:
        epoch += 1
        for i in range(0, len(X)):
            if count == len(X):
                temp = np.array([[w], [b]])
                return temp
            a = np.dot(w, X[i]) + b
            if Y[i] * a <= 0:
                count = 0
                scalar = Y[i] * X[i]
                w = w + scalar
                b = b + Y[i]
            count += 1


def perceptron_test(X_test, Y_test, w, b):
    w = w[0]
    Y_test = np.array(Y_test).flatten()
    Y_test = transform_labels(Y_test)
    print("testing")
    print(w)
    print(b)
    a = [1 if np.dot(w, X_test[i]) + b > 0 else -1 for i in range(0, len(X_test))]
    print(a)
    correct = 0
    total = 0
    for i in range(0, len(Y_test)):
        if a[i] == Y_test[i]:
            correct += 1
        total += 1
    print(correct / total)
    return correct / total


def transform_labels(Y):
    test = [1 if x == 1 else -1 for x in Y]
    return test


def dot(W, X):
    inner = 0
    for i in range(0, len(X)):
        inner += W[i] * X[i]
    return inner
