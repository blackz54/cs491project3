import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import perceptron as pt
import neural_network as nn


X = np.array([[-2, 1], [1, 1], [1.5, -0.5], [-2, -1], [-1, -1.5], [2, -2]])
Y = np.array([[0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0]])
b = nn.build_model(X, Y, 500)
>>>>>>> ea6a521a7b1109c28a1f6c6a3b8be8b274d6dcc8
