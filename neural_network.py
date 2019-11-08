import numpy as np
import sklearn as sk
import perceptron as pt
import matplotlib.pyplot as plt


# Function to calculate loss of the current version of the neural network. Computes probability distribution matrix and then computes the loss
# by calculating the summation of the real label multiplied by the log of the probability distribution for each sample. Returns the normalized loss.
def calculate_loss(model, X, y):
    yhat, a, h = zip(*[calculate_yhat(model, X[i]) for i in range(len(X))])
    loss = np.sum([y[i] * np.log(yhat[i]) for i in range(len(X))])
    loss = loss / (-1 * len(X))
    return loss


# Helper function to return a 0 or a 1 based on whichever index of the label input has a greater probability distribution.
def transform_probility_distribution(y):
    return 0 if y[0,0] > 0.5 else 1


# Helper function to compute probability distribution yhat, a, and h variables. 
def calculate_yhat(model, x):
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']

    a = np.add(np.matmul(x, W1),b1)
    h = np.tanh(a)

    z = np.add(np.matmul(h, W2),b2)
    yhat = softmax(z)
    return yhat, a, h


# Function to predict the binary output of a sample based on the current version of our neural network.
def predict(model, x):
    yhat, a, h = calculate_yhat(model, x)
    return transform_probility_distribution(yhat)


# Remove user input
def transform_labels(y):
    return [[0, 1] if y[i] == 1 else [1, 0] for i in range(len(y))]


# Function to develop the two layer neural network. Performs forward propagation and then backpropagation according to project specifications
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    if y[0] == 0 or y[0] == 1:
        y = transform_labels(y)
    W1 = init_weights(len(X[0]), nn_hdim)
    b1 = init_bias(nn_hdim)
    W2 = init_weights(nn_hdim, len(X[0]))
    b2 = init_bias(len(X[0]))
    learning_rate = 0.01  # ada
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    index = 0

    for itr in range(0, num_passes):
        yhat, a, h = calculate_yhat(model, X[index])
        # compute the derivatives for gradient descent
        dL_dyhat = yhat - y[index]
        dL_da = np.multiply((1 - np.square(h)), np.matmul(dL_dyhat, W2.transpose()))

        dL_dW2 = np.matmul(h.transpose(),dL_dyhat)
        dL_db2 = dL_dyhat

        dL_dW1 = np.matmul(np.array([X[index]]).transpose(),dL_da)
        dL_db1 = dL_da

        W1 = W1 - learning_rate * dL_dW1
        b1 = b1 - learning_rate * dL_db1
        W2 = W2 - learning_rate * dL_dW2
        b2 = b2 - learning_rate * dL_db2

        model['W1'] = W1
        model['b1'] = b1
        model['W2'] = W2
        model['b2'] = b2

        if print_loss and itr % 1000 == 0:
            error = calculate_loss(model, X, y)
            print("loss: " + str(error))
        index += 1
        if index == len(X):
            index = 0
    return model

# Function to generate a random number, mostly between -0.1 and 0.1
def generate_random_number():
    temp = np.random.randint(-1, 1)
    if temp is not 0:
        denom = np.random.randint(-100, 100)
        if denom is not 0:
            return np.around(temp / denom, 5)
    return generate_random_number()


# Function to calculate yhat prediction
def softmax(x):
    ex = np.exp(x)
    return np.true_divide(ex,ex.sum())


# Function to initialize bias variables
def init_bias(k):
    # b = [np.random.randint(-1,1) for j in range(k)]
    b = [generate_random_number() for j in range(k)]
    return np.array([b])


# Function to initialize weights for x1 and x2 samples
def init_weights(d, k):
    w = [[generate_random_number() for j in range(k)] for i in range(d)]
    return np.array(w)


def plot_decision_boundary(pred_func, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)