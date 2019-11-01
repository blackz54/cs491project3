import numpy as np
import sklearn as sk
import perceptron as pt


def calculate_loss(model, X, y):
    test = 0


def predict(model, x):
    test = 0


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    test = 0


def generate_random_number():
    temp = np.random.randint(-1, 1)
    if temp is not 0:
        denom = np.random.randint(-1000, 1000)
        return temp / denom
    else:
        return generate_random_number()


def init_bias(k):
    b = [generate_random_number() for j in range(k)]
    return b


def initialize_weights(d, k):
    w = [[np.random.randint(-10, 10) for j in range(k)] for i in range(d)]
    return w


def calc_h(x, w, b):
    test = 0


def calc_yhat(h, w, b):
    test = 0
    