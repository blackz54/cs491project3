import numpy as np
import sklearn as sk
import perceptron as pt
import matplotlib.pyplot as plt

def calculate_loss(model, X, y):
    yhat, a, h = zip(*[calculate_yhat(model, X[i]) for i in range(len(X))])
    loss = np.sum([y[i] * np.log(yhat[i]) for i in range(len(X))])
    loss = loss / (-1 * len(X))
    return loss


def transform_probility_distribution(y):
    return 0 if y[0] > 0.5 else 1


def calculate_yhat(model, x):
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']

    # print("x: " + str(x))
    # print("W1: " + str(W1))
    # print("b1: " + str(b1))
    a = [np.dot(x, W1[i]) + b1[i] for i in range(len(b1))]
    h = np.tanh(a)
    #print("a: " + str(a))
    #print("len of a: " + str(len(a)))
    # print("W2: " + str(W2))
    # print("b2: " + str(b2))
    # print("h: " + str(h))
    # print(type(h))
    # print(type(h[0]))

    z = [np.dot(h, W2[i]) + b2[i] for i in range(len(b2))]
   # print("z: " + str(z))
    y = softmax(z)
    #print("y: " + str(y))
    return y, a, h


def predict(model, x):
    #print(x)
    yhat, a, h = calculate_yhat(model, x)
    return transform_probility_distribution(yhat)


# Remove user input
def transform_labels(y):
    return [[0, 1] if y[i] == 1 else [1, 0] for i in range(len(y))]


def transpose(h):
    h_transpose = [[h[i]] for i in range(len(h))]
    return h_transpose


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    if y[0] == 0 or y[0] == 1:
        y = transform_labels(y)
    #y = transform_labels(y)
    W1 = init_weights(len(X[0]), nn_hdim)
    b1 = init_bias(nn_hdim)
    W2 = init_weights(nn_hdim, len(X[0]))
    b2 = init_bias(len(X[0]))
    learning_rate = 0.01  # ada
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    # for itr in range(0, num_passes):
    index = 0
    for itr in range(0, num_passes):
        # y = calculate_loss(model, X, y)
        # print("y: " + str(y))
        yhat, a, h = calculate_yhat(model, X[index])
        # compute the derivatives for gradient descent
        # print(a)
        # print(1 - np.tanh(h))
        dL_dyhat = yhat - y[index]
        #print("dldy: " + str(dL_dyhat))
        dL_db2 = dL_dyhat
        hT = transpose(h)
        #print("hT: " + str(hT))
        dL_dW2 = np.array(hT * dL_dyhat).transpose()
        dL_da = np.multiply((1 - np.tanh(h)), np.dot(dL_dyhat, W2))
        dL_db2 = dL_dyhat
        dL_db1 = dL_da
        dL_dW1 = np.array(transpose(X[index]) * dL_da).transpose()
        # print("dlda: " + str(dL_da))
        # print("dldb1: " + str(dL_db1))
        # print("dldb2: " + str(dL_db2))
        # print("dldW1: ")
        # print(str(dL_dW1))
        # print("dldW2: " + str(dL_dW2))
        W1 = W1 - learning_rate * dL_dW1
        b1 = b1 - learning_rate * dL_db1
        W2 = W2 - learning_rate * dL_dW2
        b2 = b2 - learning_rate * dL_db2
        #print("W1:" + str(W1))
        #print("W2: " + str(W2))
        #print("b1:" + str(b1))
        #print("b2: " + str(b2))
        # a = [np.dot(X[i], W1[i]) + b1[i] for i in range(len(b1))]
        # h = np.tanh(a)
        # z = [np.dot(h, W2[i]) + b2[i] for i in range(len(b2))]
        # yhat = softmax(z)
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
    return ex / ex.sum(axis=0)


# Function to initialize bias variables
def init_bias(k):
    # b = [np.random.randint(-1,1) for j in range(k)]
    b = [generate_random_number() for j in range(k)]
    return b


# Function to initialize weights for x1 and x2 samples
def init_weights(d, k):
    w = [[generate_random_number() for j in range(d)] for i in range(k)]
    return w


def plot_decision_boundary(pred_func, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)