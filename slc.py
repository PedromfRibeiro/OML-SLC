import numpy as np
import matplotlib.pyplot as plt
from random import randint, random
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
    epsi = 1.e-12
    sum = 0
    for n in range(N):
        prediction = predict(X[n], J, w, b, v, c)
        # verify if the value of the prediction is in [1e-12, 1-1e-12]
        if prediction < epsi: 
            prediction = epsi
        if prediction > 1 - epsi:
            prediction = 1 - epsi
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
    #print('a')
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
    delta = prediction - y
    Gw = delta * np.matmul(matrix1, matrix2) # w's gradient
    Gb = delta * np.dot(matrix1, v)          # b's gradient
    Gv = delta * h                           # v's gradient
    Gc = delta                               # c's gradient
    '''
    delta = prediction - y
    diagonal = np.multiply(h, (1 - h))
    Gb = delta * np.multiply(v, diagonal)
    Gw = np.tensordot(Gb, x, axes=0)
    Gv = delta * h
    Gc = delta    
    '''
    return Gw, Gb, Gv, Gc

'''
Update w, b, v, c
'''
def update(x, y, J, eta, w, b, v, c): 
    # calculate the gradients
    h = get_h(x, w, b) 
    prediction = predict(x, J, w, b, v, c)
    Gw, Gb, Gv, Gc = gradient(x, y, v, h, prediction)
    # update the first layer
    w = w - (eta * Gw) # w (t + 1)
    b = b - (eta * Gb) # b (t + 1)
    # update the second layer
    v = v - (eta * Gv) # v (t + 1)
    c = c - (eta * Gc) # c (t + 1)
    return w, b, v, c

'''
Run shallow logistic classifier
'''
def run_slc(X, Y, N, J, subloop, eta, max_iteration, w, b, v, c, errors):
    #epsi = 10e-3
    epsi = 0
    iteration = 0
    while (errors[-1] > epsi):
        for j in range(subloop):
            # choose random data from dataset
            m = randint(0, N - 1)
            x = X[m]
            y = Y[m]
            # update w, b, v, c from (t) to (t + 1)
            w, b, v, c = update(x, y, J, eta, w, b, v, c)
            # calculate error
        error = cost(X, Y, N, w, b, v, c)
        errors.append(error)
        print('number of iterations = %d, cost = %f, eta = %e\r' %(iteration, errors[-1], eta), end='')
        iteration += subloop
        if (iteration > max_iteration):
            break
    return w, b, v, c, errors
 
def plot_error(err):
    plt.plot(range(len(err)), err, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    plt.ylim([0,5])
    plt.show()
    return 

def confusion(Xeval,Yeval,N, J, w, b, v, c):
    C=np.zeros([2,2])
    for n in range(N):
        y = predict(Xeval[n], J, w, b, v, c)
        if(y<0.5 and Yeval[n]<0.5): C[0,0]=C[0,0]+1
        if(y>0.5 and Yeval[n]>0.5): C[1,1]=C[1,1]+1
        if(y<0.5 and Yeval[n]>0.5): C[1,0]=C[1,0]+1
        if(y>0.5 and Yeval[n]<0.5): C[0,1]=C[0,1]+1
    return C
       
'''
Main execution
'''

# load data from dataset
#dataset = './Data/XOR.txt'
#dataset = './Data/line600.txt'      # almost linearly separable (J = 2)
dataset = './Data/rectangle600.txt' # it's harder to separate linearly (J = 10)
#dataset = './Data/square_circle.txt' # linearly separable (J = 1)
N, n_row, n_col, data = load_data(dataset)
I = n_row * n_col

# shuffle dataset
np.random.shuffle(data)

# split data into training and testing sets
N_train = int(N * 0.8) 
X_train = data[:N_train, :-1]
Y_train = data[:N_train, -1]
N_test = N - N_train
X_test = data[N_train:N, :-1]
Y_test = data[N_train:N, -1]

# replace labels -1 for 0
np.place(Y_train, Y_train != 1, [0])
np.place(Y_test, Y_test!=1, [0])

J = 10 # number of neurons in the hidden layer 

# initialize w, b, v and c with random values
w = np.random.randn(J, I)
b = np.random.randn(J)
v = np.random.rand(J)
c = np.random.rand(1)

eta = 1 # learning rate

Nbiter = 3000 # number of iterations
subloop = 300 # only calculate the cost in each 100 iterations

# initialize errors' list
errors = []
errors.append(cost(X_train, Y_train, N_train, w, b, v, c))

w, b, v, c, errors = run_slc(X_train, Y_train, N_train, J, subloop, eta, Nbiter, w, b, v, c, errors); print("\n"); eta = 0.5 * eta
w, b, v, c, errors = run_slc(X_train, Y_train, N_train, J, subloop, eta, Nbiter, w, b, v, c, errors); print("\n"); eta = 0.5 * eta
w, b, v, c, errors = run_slc(X_train, Y_train, N_train, J, subloop, eta, Nbiter, w, b, v, c, errors); print("\n"); eta = 0.5 * eta
w, b, v, c, errors = run_slc(X_train, Y_train, N_train, J, subloop, eta, Nbiter, w, b, v, c, errors); print("\n")

plot_error(errors)

print('in-samples error = %f ' % (cost(X_train, Y_train, N_train, w, b, v, c)))
# calculate and show confusion matrix
C = confusion(X_train, Y_train, N_train, J, w, b, v, c)
print(C)

print('out-samples error=%f' % (cost(X_test, Y_test, N_test, w, b, v, c)))
# calculate and show confusion matrix
C =confusion(X_test, Y_test, N_test, J, w, b, v, c)
print(C)
