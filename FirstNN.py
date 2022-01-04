from LRNN.utils import *
from Fashion.utils import *
import matplotlib.pyplot as plt

"""
Label	Description
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
"""
newY =[]
newYt=[]
X,Y = load_mnist("Fashion/dataset")
for i in range(0,Y.shape[0]):
    if Y[i] in [5,7,9]:
        newY.append(1)
    else:
        newY.append(0)   

Xt,Yt = load_mnist("Fashion/dataset",kind="t10k")
for i in range(0,Yt.shape[0]):
    if Yt[i]in [5,7,9]:
        newYt.append(1)
    else:
        newYt.append(0) 
plt.imshow(X[0].reshape((28,28)))
train_set_x = X.T/255.
test_set_x = Xt.T/255.
test_set_y= np.array(newYt)
train_set_y= np.array(newY)

w, b = initialize_with_zeros(train_set_x.shape[0])
Y_prediction_test = predict(w, b, test_set_x)
Y_prediction_train = predict(w, b, train_set_x)
# Print train/test Errors
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))
print(Y_prediction_test)



w,b = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 500, learning_rate = 0.02, print_cost = True)


# Print train/test Errors
Y_prediction_test = predict(w, b, test_set_x)
Y_prediction_train = predict(w, b, train_set_x)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))
print(Y_prediction_test)
