import numpy as np
import sklearn as sk
import perceptron as pt


def calculate_loss(model, X, y):
    yhat = [calculate_yhat(model, X[i]) for i in range(len(X))]
    loss = np.sum([y[i] * np.log(yhat[i]) for i in range(len(X))])
    loss = loss / (-1 * len(X))
    print(loss)
    return loss

def transform_probility_distribution(y):
    return -1 if y[0] > 0.5 else 1

def calculate_yhat(model, x):
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    #print("x: " + str(x))
    #print("W1: " + str(W1))
    #print("b1: " + str(b1))
    #print("W2: " + str(W2))
    #print("b2: " + str(b2))
    a = [np.dot(x, W1[i]) + b1[i] for i in range(len(b1))]
    h = np.tanh(a)
    #print("a: " + str(a))
    #print("len of a: " + str(len(a)))
    #print("h: " + str(h))
    z = [np.dot(h, W2[i]) + b2[i] for i in range(len(b2))]
    #print("z: " + str(z))
    y = softmax(z)
    #print("y: " + str(y))
    return y

def predict(model, x):
    yhat = calculate_yhat(model, x)
    return transform_probility_distribution(yhat)

# Remove user input
def transform_labels(y):
    return [[0,1] if y[i] == 1 else [1,0] for i in range(len(y))]

def build_model(X, y, nn_hdim, num_passes=20, print_loss=False):
    y = transform_labels(y)
    W1 = init_weights(len(X[0]), nn_hdim)
    b1 = init_bias(nn_hdim)
    W2 = init_weights(nn_hdim, len(X[0]))
    b2 = init_bias(len(X[0]))
    learning_rate = 0.01 # ada
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    # for itr in range(0, num_passes):
    for itr in range(0, 1):
        G = [[0 for j in range(len(X[0]))] for i in range(nn_hdim)]
        g = [0 for i in range(nn_hdim)]
        #y = [predict(model, X[i]) for i in range(len(X))]
        y = calculate_loss(model,X,y)
        #print("y: " + str(y))

# Function to generate a random number, mostly between -0.1 and 0.1
def generate_random_number():
    temp = np.random.randint(-1, 1)
    if temp is not 0:
        denom = np.random.randint(-100, 100)
        if denom is not 0:
            return np.around(temp/denom, 5)
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