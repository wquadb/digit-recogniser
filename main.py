import os
import numpy as np
import pandas as pd
from PIL import Image as im

from matplotlib import pyplot as plt
from scipy.special import softmax


def get_data(link) -> (np.array, np.array):

    """
    making ndarray from .csv such that vectors are images
    with i-th row as j-th pixel of i-th  (1 <= i <= 784; 1 <= j <= 60.000)
    """

    file = pd.read_csv(link)

    vector = np.array(file["label"]).T
    file = file.drop("label", axis=1)

    array = np.array(file).transpose()

    return array, vector

def init_weights():

    """
    generate weights and biases
    to start the model
    """

    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return w1, w2, b1, b2

def ReLU(x):
    
    # f(x) = { x, if x > 0
    #        { 0, if x <= 0

    return np.maximum(0, x)

def deriv_ReLU(Z):
    """
    Returns boolean value True/False, 
    it is equal to 1/0 as it used in equation 
    """
    return Z > 0

def forward_propagation(A0,w1,w2,b1,b2):

    """
    func get results from train_dataset that passed 
    through the equations and led to the probabilities
    """

    Z1 = w1.dot(A0) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    
    return Z1, A1, Z2, A2

def get_accurate_vectors(learn_vector) -> np.array:

    # very difficult way to get truly correct answer vector
    # f.e. [0,1,0,0,0,0,0,0,0,0] from [1]; [0,0,0,0,0,5,0,0,0,0] from [5]

    vectors = [[0 if learn_vector[x] != i else 1 for i in range(10)] for x in range(len(learn_vector))]
    return np.array(vectors).transpose()

def backward_propagation(Y0, w1, w2, b1, b2, A0, A1, A2, Z1, Z2):
    """
    get values of falsity of incorrect answers (function loss)
    that will be used in gradient descent
    """

    dZ2 = A2 - Y0
    m = dZ2.size
    dW2 = 10 / m * dZ2.dot(A1.T)
    db2 = 10 / m * np.sum(dZ2, 1)
    dZ1 = (w2.T).dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 10 / m * dZ1.dot(A0.T)
    db1 = 10 / m * np.sum(dZ1, 1)

    # db1 and db2 are 10x1 but when we do not reshape 
    # np.info() tells: 
    # db1.shape() = (10,) | db2.shape() = (10,)
    # the it breakes with no reason (b1, b2 shapes gets (10,10) from (10,1) )
    # stack overflow suggest this article about this issue in numpy:
    # https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r

    db1 = db1.reshape(10,1)
    db2 = db2.reshape(10,1)

    return dW1, dW2, db1, db2

def update(w1, w2, b1, b2, dW1, dW2, db1, db2, learning_rate=0.1):
    
    # substract function loss result from 
    # previous weights to get better 
    # coefficients and biases

    alph = learning_rate
    w1 = w1 - alph * dW1    
    w2 = w2 - alph * dW2
    b1 = b1 - alph * db1
    b2 = b2 - alph * db2
    return w1, w2, b1, b2

def get_accuracy(A2, Y0):

    # get accuracy func were borrowed from youtube video
    # so it should be rewrited, after that we definitely 
    # will be able to be sure that it works properly

    pred = np.argmax(A2, 0) 
    return np.sum(pred == Y0) / Y0.size

def main(iterations):

    #   Gradient-descent
    # Some big issue with performing gradient, if we demonstrate 
    # weights matrix via func demonstrate, we'll see that it has
    # same pattern for each node, which can be represental mistake
    # or mistake in understanding of the model, but it should be explained

    data_learn_pixels, data_learn_answers = get_data('mnist_train.csv')

    A0, Y1 = data_learn_pixels, data_learn_answers

    w1, w2, b1, b2 = init_weights()

    Y0 = get_accurate_vectors(Y1)

    for i in range(iterations):

        Z1, A1, Z2, A2 = forward_propagation(A0=A0, w1=w1, w2=w2, b1=b1, b2=b2)

        dW1, dW2, db1, db2 = backward_propagation(Y0=Y0, w1=w1, w2=w2, b1=b1, b2=b2, A0=A0, A1=A1, A2=A2, Z1=Z1, Z2=Z2)

        w1, w2, b1, b2 = update(w1=w1, w2=w2, b1=b1, b2=b2, dW1=dW1, dW2=dW2, db1=db1, db2=db2, learning_rate=0.02)

        if i % 2 == 0:
            print("Iteration: ", i)
            print(get_accuracy(A2=A2, Y0=Y0))

    return w1, w2, b1, b2

def demonstrate(w1):
    
    # uses pyplot to demonstate result of learning nodes in first layer
    # this way has some issues to fix, so it should be rewrited as much
    # as get_accuracy function
    
    for i in w1:
        grid = list([x for x in i[y::28]] for y in range(28))
        fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})
        for ax in axs.flat:
            ax.imshow(grid, interpolation='none', cmap='viridis')
    plt.show()

def vecs_to_img(array, fromim: int, toim: int):

    # using standart library of Python 
    # script create n images from array of pixels (n = toim - fromim + 1)
    # it should be remade with matplotlib, or remake it because it awful to read

    scale = (toim - fromim + 1) * 28
    for y in range(fromim, toim + 1):
        first = []
        for i in range(28):
            first.append([array.T[x][y] for x in range(784)][i * 28 : (i + 1) * 28])
        first = np.array(first)
        sec = im.fromarray(first.astype('uint8')*255)
        sec.save(f"results{y}.jpg")
    nU = im.fromarray((np.zeros(shape=(scale, scale))).astype('uint8')*255)
    k, b = 0, 0
    for y in range(fromim, toim + 1):
        if np.random.randint(0,1) == 0:
            k += 28
        else:
            b += 28
        img = im.open(f"results{y}.jpg")
        nU.paste(img,(k,b))
        os.remove(f"results{y}.jpg")
    nU.show()

if __name__ == "__main__":

    w1, w2, b1, b2 = main(iterations=10)

    demonstrate(w1)

    # another big issue is that no matter of amount of
    # operations, matplotlib demontrates exactly same
    # weights (unfortunately, it may be not represental mistake)

    # main task for understanding our model is to surely convince
    # ourselves that all the computations work in the proper way
    # forward propagation works quite simplyand correct so 
    # gradient descent and backwards propagation require
    # much more effort to prove
