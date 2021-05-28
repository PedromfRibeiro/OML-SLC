import numpy as np
import csv
from random import randint

from numpy.core.fromnumeric import diagonal

'''
Load data from file
'''
def load_data(filename):    
    file = open(filename, 'r') 
    tmp_str = file.readline()
    tmp_arr = tmp_str[:-1].split(' ')
    N = int(tmp_arr[0])
    n_row = int(tmp_arr[1])
    n_col = int(tmp_arr[2])
    print('N=%d, row=%d, col=%d' %(N,n_row,n_col))
    data = np.zeros([N, n_row * n_col + 1])
    for n in range(N):
        tmp_str = file.readline()
        tmp_arr = tmp_str[:-1].split(' ')       
        for i in range(n_row * n_col + 1):
            data[n][i] = int(tmp_arr[i])
    file.close() 
    return N, n_row, n_col, data

'''
Sigmoid function
'''
def sigmoid(s):  
    large=30
    if s<-large: s=-large
    if s>large: s=large
    return (1 / (1 + np.exp(-s)))

'''
Cost funtion
'''
def cost(X, Y, N, w, b, v, c):
    sum = 0
    for n in range(N):
        prediction = predict(X[n], J, w, b, v, c)
        sum += Y[n] * np.log(prediction) + (1 - Y[n]) * np.log(1 - prediction)
    E = - sum / N
    return E

'''
Predict label
'''
def predict(x, J, w, b, v, c):
    # x -> w * x + b -> s
    # s -> sigmoid(s) -> h
    
    # h -> c + v.T * h -> z
    sum = 0
    for j in range(J):
        sum += v[j] * h[j]
    z = c + sum
    y = sigmoid(z)
    return y


def gradient(X, Y, v, h, m, prediction):
    func1 = lambda a: a * (1 - a)
    diagonal = np.array([func1(h_m) for h_m in h])
    matrix1 = np.diag(diagonal) # J x J
    matrix2 = np.tile(X[m], (J, 1))    
    matrix2 = np.transpose(matrix2) * v
    matrix2 = np.transpose(matrix2) # J x I
    Gw = (prediction - Y[m]) * np.matmul(matrix1, matrix2) # w's gradient
    Gb = (prediction - Y[m]) * (matrix1 * v)               # b's gradient
    Gv = (prediction - Y[m]) * h                           # v's gradient
    Gc = prediction - Y[m]                                 # c's gradient
    return Gw, Gb, Gv, Gc

'''
'''
def update(X, Y, J, m, eta, w, b, v, c, h):
    prediction = predict(J, v, c, h)
    Gw, Gb, Gv, Gc = gradient(X, Y, N, w, b, v, c, h, m, prediction)
    w = w - eta * Gw # w (t + 1)
    b = b - eta * Gb # b (t + 1)
    v = v - eta * Gv # v (t + 1)
    c = c - eta * Gc # c (t + 1)
    return w, b, v, c

'''
Run shallow logistic classifier
'''
def run_slc(X, Y, N, J, eta, max_iteration, w, b, v, c, h, errors):
    epsi = 10e-3
    iteration = 0
    while (errors[-1] > epsi):
        iteration += 1
        if (iteration > max_iteration):
            break
        # choose random data from dataset
        m = randint(0, N - 1)
        # update w, b, v, c from (t) to (t + 1)
        w, b, v, c = update(X, Y, J, m, eta, w, b, v, c, h)
        error = cost(X, Y, N, w, b, v, c)
        errors.append(error)
    return w, b, v, c, errors
        
'''
Main execution
'''

# load data from dataset
N, n_row, n_col,data = load_data('./Data/XOR.txt') # and -> j = 1, xor -> j = 2
N = int(N * 1.0)
I = n_row * n_col
X = data[:N, :-1] 
Y = data[:N, -1]

J = 3 # number of neurons in the hidden layer

# initialize w, b, v, c
w = np.ones((J, I))
b = np.ones(J)
v = np.ones(J)
c = 1
h = np.ones(J) # hidden layer's weigths

eta = 0.1 # learning rate

# initialize errors' list
errors = []
errors.append(cost(X, Y, N, w, b, v, c))

run_slc(X, Y, N, J, eta, 300, w, b, v, c, h, errors)