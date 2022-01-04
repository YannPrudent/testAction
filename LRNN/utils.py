import numpy as np

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w, b

def initialize_with_rnd(dim):
    w = np.random.rand(dim,1)*0.1
    b = 0
    return w, b

def compute_cost(A,Y,m):
    cost = -(np.dot(Y,np.log(A).T)+np.dot((1-Y),np.log(1-A).T))/m
    return cost


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0,i]>0.5:
            Y_prediction[0,i]=1
        else:
             Y_prediction[0,i]=0
       
    return Y_prediction


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost=compute_cost(A,Y,m) 
    dZ=A-Y
    dw = np.dot(X,dZ.T)/m
    db = np.sum(dZ)/m
    return dw, db, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    for i in range(num_iterations):
        dw, db, cost = propagate(w, b, X,Y)
        w = w-learning_rate*dw
        b = b-learning_rate*db
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return w,b



def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    print(X_train.shape[0])
    w, b = initialize_with_zeros(X_train.shape[0])
    # Gradient descent
    w, b = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
   
   
    return w,b



