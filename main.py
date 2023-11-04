import os
import numpy as np
import pandas as pd
from PIL import Image as im
from scipy.special import softmax

#TO FIX
def vecs_to_img(array, fromim: int, toim: int):
    scale = (toim - fromim + 1) * 28
    c = array
    for y in range(fromim, toim + 1):
        first = []
        for i in range(28):
            a = i
            b = i + 1
            first.append([c.T[x][y] for x in range(784)][a * 28 : b * 28])
        first = np.array(first)
        sec = im.fromarray(first.astype('uint8')*255)
        sec.save(f"results{y}.jpg")
    nU = im.fromarray((np.zeros(shape=(scale, scale))).astype('uint8')*255)
    k, b = 0, 0
    for y in range(fromim, toim + 1):
        if k < scale:
            k += 28
        else:
            k = 0
            if b < scale:
                b += 28
            else:
                b = 0
                break
        img = im.open(f"results{y}.jpg")
        nU.paste(img,(k,b))
        os.remove(f"results{y}.jpg")
    nU.show()
    
def get_data(link) -> (np.array, np.array):

    """
    making ndarray from .csv
    such that vectors are images
    with i-th row as i-th pixel of image
    """

    file = pd.read_csv(link)

    vector = np.array(file["label"]).T
    file = file.drop("label", axis=1)

    array = np.array(file)

    return array.transpose(), vector

def init_weights():

    """
    randomly generated weights and biases
    for 2 layer neural network
    """

    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return w1, w2, b1, b2

def ReLU(x):
    return np.maximum(0, x)

def deriv_ReLU(Z):
    return Z > 0

def forward_propagation(A0,w1,w2,b1,b2):

    Z1 = w1.dot(A0) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    
    return Z1, A1, Z2, A2

def get_accurate_vectors(learn_vector) -> np.array:
    vectors = [[0 if learn_vector[x] != i else 1 for i in range(10)] for x in range(len(learn_vector))]
    return np.array(vectors).transpose()

def backward_propagation(Y0, w1, w2, b1, b2, A0, A1, A2, Z1, Z2):
    dZ2 = A2 - Y0
    m = dZ2.size
    dW2 = 10 / m * dZ2.dot(A1.T)
    db2 = 10 / m * np.sum(dZ2, 1)
    dZ1 = (w2.T).dot(dZ2) * deriv_ReLU(Z1)
    m = dZ1.size
    dW1 = 10 / m * dZ1.dot(A0.T)
    db1 = 10 / m * np.sum(dZ1, 1)
    return dW1, dW2, db1, db2

def update(w1, w2, b1, b2, dW1, dW2, db1, db2, learning_rate=0.1):
    alph = learning_rate
    w1 = w1 - alph * dW1
    b1 = b1 - alph * db1
    w2 = w2 - alph * dW2
    b2 = b2 - alph * db2
    return w1, w2, b1, b2

def get_accuracy(A2, Y0):
    pred = np.argmax(A2, 0) 
    return np.sum(pred == Y0) / Y0.size


if __name__ == "__main__":

    data_learn_pixels, data_learn_answers = get_data('mnist_train.csv')

    vecs_to_img(data_learn_pixels.T, 59998, 59999)

    data_learn_answers







#GRADIENT DESCENT
def main():

    data_learn_pixels, data_learn_answers = get_data('mnist_train.csv')

    A0, Y1 = data_learn_pixels, data_learn_answers

    w1, w2, b1, b2 = init_weights()

    Y0 = get_accurate_vectors(Y1)

    for i in range(100):

        Z1, A1, Z2, A2 = forward_propagation(A0=A0, w1=w1, w2=w2, b1=b1, b2=b2)

        dW1, dW2, db1, db2 = backward_propagation(Y0=Y0, w1=w1, w2=w2, b1=b1, b2=b2, A0=A0, A1=A1, A2=A2, Z1=Z1, Z2=Z2)

        w1, w2, b1, b2 = update(w1=w1, w2=w2, b1=b1, b2=b2, dW1=dW1, dW2=dW2, db1=db1, db2=db2, learning_rate=0.1)

        if i % 10 == 0:
            print("Iteration: ", i)
            print(get_accuracy(A2=A2, Y0=Y0))

    return w1, w2, b1, b2

