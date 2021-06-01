import numpy as np
import matplotlib.pyplot as plt
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

def normalization(w):
    return(w/np.linalg.norm(w,2))
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
def gradient(x, y, v, h, prediction, eta):
    func = lambda a: a * (1 - a)
    diagonal = np.array([func(h_m) for h_m in h])
    matrix1 = np.diag(diagonal) # J x J
    matrix2 = np.tile(x, (J, 1))    
    matrix2 = np.transpose(matrix2) * v
    matrix2 = np.transpose(matrix2) # J x I
    delta = prediction - y
    #delta=np.sign(delta)*min(0.1,np.abs(delta));
    #delta = delta* eta
    Gw = (delta) * np.matmul(matrix1, matrix2) # w's gradient
    Gb = (delta) * np.dot(matrix1, v)          # b's gradient
    Gv = (delta) * h                           # v's gradient
    Gc = delta                             # c's gradient
    return Gw, Gb, Gv, Gc

'''
Update w, b, v, c
'''
def update(x, y, J, eta, w, b, v, c): 
    h = get_h(x, w, b) 
    prediction = predict(x, J, w, b, v, c)
    Gw, Gb, Gv, Gc = gradient(x, y, v, h, prediction, eta)
    w = w - (eta * Gw) # w (t + 1)
    b = b - (eta * Gb) # b (t + 1)
    v = v - (eta * Gv) # v (t + 1)
    c = c - (eta * Gc) # c (t + 1)
    return w, b, v, c

'''
Run shallow logistic classifier
'''
def run_slc(X, Y, N, J, subloop, eta, max_iteration, w, b, v, c, errors):
    epsi = 10e-3
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
        print('iter %d, cost=%f, eta=%e     \r' %(iteration,errors[-1],eta),end='')
        iteration += 1
        if (iteration > max_iteration):break
    return w, b, v, c, errors

def plot_data(row,col,n_row,n_col,data):
    fig=plt.figure(figsize=(row,col))
    for n in range(1, row*col +1):
        img=np.reshape(data[n-1][:-1],(n_row,n_col))
        fig.add_subplot(row, col, n)
        plt.imshow(img,interpolation='none',cmap='binary')
    plt.show()
'''    
def plot_tagged_data(row,col,n_row,n_col,X,Y,ew): 
    fig=plt.figure(figsize=(row,col))
    for n in range(row*col):
        img=np.reshape(X[n],(n_row,n_col))
        fig.add_subplot(row, col, n+1)
        #if(Y[n]>0):#exact case
        if(predict(X[n],ew)>0.5):
            plt.imshow(img,interpolation='none',cmap='RdPu')
        else:
            plt.imshow(img,interpolation='none',cmap='cool')               
    plt.show()
'''    
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
N, n_row, n_col,data = load_data('./Data/XOR.txt')
#N, n_row, n_col,data = load_data('./Data/line1500.txt')
N = int(N * 1.0)
I = n_row * n_col
X = data[:N, :-1] 
Y = data[:N, -1]
#eta=1.0;
Nbiter=1200;
subloop=200;
# J = 1 for AND, J = 2 for XOR, J = 3 for line600
J = 2 # number of neurons in the hidden layer

# initialize w, b, v, c
w = np.ones((J, I))
#w=normalization(w)*0;
b = np.ones(J)
#b=normalization(b)*0;
v = np.ones(J)
#v=normalization(v)*0;

c = 1
#c=normalization(c)*0;
eta = 1.0  # learning rate

# initialize errors' list
errors = []
errors.append(cost(X, Y, N, w, b, v, c))

w, b, v, c, errors = run_slc(X, Y, N, J, subloop, eta, Nbiter, w, b, v, c, errors);print("\n");eta=0.7*eta;
w, b, v, c, errors = run_slc(X, Y, N, J, subloop, eta, Nbiter, w, b, v, c, errors);print("\n");eta=0.5*eta;
w, b, v, c, errors = run_slc(X, Y, N, J, subloop, eta, Nbiter, w, b, v, c, errors);print("\n");eta=0.3*eta;
w, b, v, c, errors = run_slc(X, Y, N, J, subloop, eta, Nbiter, w, b, v, c, errors);print("\n");eta=0.4*eta;
#w, b, v, c, errors = run_slc(X, Y, N, J, subloop, eta, Nbiter, w, b, v, c, errors)
plot_error(errors)

print('in-samples error = %f ' % (cost(X, Y, N, w, b, v, c)))
#print(ew)
C = confusion(X, Y, N, J, w, b, v, c)
print(C)
