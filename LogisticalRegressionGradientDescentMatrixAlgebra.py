import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Logistic Regression is bother faster and easier to do with matrix algebra")

X = np.asarray([
    [0.50],[0.75],[1.00],[1.25],[1.50],[1.75],[1.75],
    [2.00],[2.25],[2.50],[2.75],[3.00],[3.25],[3.50],
    [4.00],[4.25],[4.50],[4.75],[5.00],[5.50]])

ones = np.ones(X.shape)
X = np.hstack([ones, X])

Y = np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]).reshape([-1, 1])
Theta = np.array([[0], [0]]) #just trying to translate whats happening

#Lets define functions to maker everything easier!
#standard sigmoid function
def sigmoid(a):
    return 1.0 / (1 + np.exp(-a))

#np.natmul is just matrix multiplication
def cost(x, y, theta):
    m = x.shape[0]
    h = sigmoid(np.matmul(x, theta))
    cost = (np.matmul(-y.T, np.log(h)) - np.matmul((1 -y.T), np.log(1 - h)))/m
    return cost


n_iterations = 10200
learning_rate = 0.5

for i in range(n_iterations):
    m = X.shape[0]
    h = sigmoid(np.matmul(X, Theta))
    grad = np.matmul(X.T, (h - Y)) / m;
    Theta = Theta - learning_rate * grad


print(Theta)
print("This executes much faster using matrix algebra, lets plot it again")
print("This script does exactly the same thing as using gradient descent without matrix algebra")

xvals=np.linspace(0,6,100)
print(xvals)
yvals=[]
for hh in xvals:
    print(1.0 / (1 + np.exp(-(Theta[0]+Theta[1]*hh))))
    yvals.append((1.0 / (1 + np.exp(-(Theta[0]+Theta[1]*hh)))))

plt.title("Logistic regression and plotting of the model with the data points",fontsize=14)
plt.ylabel("Probability",fontsize=14)
plt.xlabel("X-Value",fontsize=14)
plt.scatter(X[:,1],Y)
plt.plot(xvals,yvals,'r')
plt.show()
