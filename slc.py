import numpy as np
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
Cost/Loss funtion
'''
def cost(X, Y, N, w, b, v, c):
    sum = 0
    for n in range(N):
        prediction = predict(X[n], J, w, b, v, c)
        sum += Y[n] * np.log(prediction) + (1 - Y[n]) * np.log(1 - prediction)
    E = - sum / N
    return E

'''
Calculate h
'''
def get_h(x, w, b):
    # x, w, b -> b + w * x -> s
    s = b + np.dot(w, x)
    # s -> s.map(sigmoid) -> h
    func = lambda a: sigmoid(a)
    h = np.array([func(s_j) for s_j in s])
    return h

'''
Predict label
'''
def predict(x, J, w, b, v, c):
    h = get_h(x, w, b)
    # c, v, h -> c + v.T * h -> z
    sum = 0
    for j in range(J):
        sum += v[j] * h[j]
    z = c + sum
    # prediction
    y = sigmoid(z)
    return y

'''
Calculate the gradients for w, b, v, c
'''
def gradient(x, y, v, h, prediction):
    func = lambda a: a * (1 - a)
    diagonal = np.array([func(h_m) for h_m in h])
    matrix1 = np.diag(diagonal) # J x J
    matrix2 = np.tile(x, (J, 1))    
    matrix2 = np.transpose(matrix2) * v
    matrix2 = np.transpose(matrix2) # J x I
    Gw = (prediction - y) * np.matmul(matrix1, matrix2) # w's gradient
    Gb = (prediction - y) * np.dot(matrix1, v)          # b's gradient
    Gv = (prediction - y) * h                           # v's gradient
    Gc = prediction - y                                 # c's gradient
    return Gw, Gb, Gv, Gc

'''
Update w, b, v, c
'''
def update(x, y, J, eta, w, b, v, c): 
    h = get_h(x, w, b) 
    prediction = predict(x, J, w, b, v, c)
    Gw, Gb, Gv, Gc = gradient(x, y, v, h, prediction)
    w = w - eta * Gw # w (t + 1)
    b = b - eta * Gb # b (t + 1)
    v = v - eta * Gv # v (t + 1)
    c = c - eta * Gc # c (t + 1)
    return w, b, v, c

'''
Run shallow logistic classifier
'''
def run_slc(X, Y, N, J, eta, max_iteration, w, b, v, c, errors):
    epsi = 10e-3
    iteration = 0
    while (errors[-1] > epsi):
        iteration += 1
        if (iteration > max_iteration):
            break
        # choose random data from dataset
        m = randint(0, N - 1)
        x = X[m]
        y = Y[m]
        # update w, b, v, c from (t) to (t + 1)
        w, b, v, c = update(x, y, J, eta, w, b, v, c)
        # calculate error
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

eta = 0.1 # learning rate

# initialize errors' list
errors = []
errors.append(cost(X, Y, N, w, b, v, c))

run_slc(X, Y, N, J, eta, 5, w, b, v, c, errors)
print(errors)