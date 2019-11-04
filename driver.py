import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import perceptron as pt
import neural_network as nn


X = np.array([[-2, 1], [1, 1], [1.5, -0.5], [-2, -1], [-1, -1.5], [2, -2]])
Y = np.array([[1], [1], [1], [-1], [-1], [-1]])
b = nn.build_model(X, Y, 4)