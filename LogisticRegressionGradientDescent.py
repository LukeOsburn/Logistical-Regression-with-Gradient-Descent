import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#lets learn how to do logistic regression on a simple dataset
#Using gradient descent

#This is our data
X = np.asarray([
    [0.50],[0.75],[1.00],[1.25],[1.50],[1.75],[1.75],
    [2.00],[2.25],[2.50],[2.75],[3.00],[3.25],[3.50],
    [4.00],[4.25],[4.50],[4.75],[5.00],[5.50]])



Y = np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]).reshape([-1, 1])



#So what are we trying to do?
#We are going to create a model that predicts a binary Y value, 0 or 1, based only on the X value

#Our hypothesis function:

#H(ThetaX)=1/((1+e^-(X0*theta0+X1*Theta1)))
#our X data are the X1 values so we need to add a column of X0 data that will just be ones
#this is the same as in linear regression where we multiplied the y-intercept by 1


ones = np.ones(X.shape)
X = np.hstack([ones, X])

#What does our data look like now
print("Our input data looks like this now")
print(X)




print("Our hypothesis funciton is essentially the sigmoid function, where the input is, X0Theta0+X1Theta1+X2Theta2+Xn*Thetan")
def sigmoid(a):
    return 1.0 / (1 + np.exp(-a))

#Lets initiate our starting theta values
theta0=0
theta1=0
SXDT=0

#Our cost function, Sum of data points: -log(H(ThetaX))y-log(1-H(ThetaX))(1-y)
#We need the partial derivatives of this cost fuction, its a nightmare, just take the results
#Derivative of Cost Function: H(ThetaX-y)(x)


#This is for illustrative purposes
#describing what happens on the first iteration only
#for i in range(0,len(Y),1):
for i in range(0,1,1): #lets just do the first data point
    print("This is sigmoid of thetas * x-values")
    print(sigmoid((X[i,0]*theta0)+X[i,1]*theta1))
    SXDT=sigmoid((X[i,0]*theta0)+X[i,1]*theta1)
    print("This is the new theta 0 value component from the ith data point")
    print((SXDT-Y[i])*(X[i,0]))
    print("This is the new theta 1 value component from the ith data point")
    print((SXDT-Y[i])*(X[i,1]))
    print("The individual components need to be added together to get the new theta0 and theta1 values from the iteration")


#Choose number of iterations
iterations=400

#initial thetas
theta0=0
theta1=0
learningrate=0.5

#Now we have everything to do a gradient descent and find our optimal theta values, lets do it

for iter in range(0,iterations,1):

    #print("thetas")
    #print(theta0)
    #print(theta1)
    theta0adding=0
    theta1adding=0
    for i in range(0,len(Y),1):
        #these are the theta values for each data point within 1 iteration
        theta0adding=theta0adding+((sigmoid((X[i,0]*theta0)+X[i,1]*theta1)-Y[i])*X[i,0])
        theta1adding=theta1adding+((sigmoid((X[i,0]*theta0)+X[i,1]*theta1)-Y[i])*X[i,1])
    theta0=theta0-(learningrate*(theta0adding/len(Y)))
    theta1=theta1-(learningrate*(theta1adding/len(Y)))


print("These are our theta values")
print(theta0)
print(theta1)
print("Increase number of iterations to improve accuracy")


theta0=round(theta0, 2)
theta1=round(theta1, 2)

#okay, so this is our final model!
print("This is our final model, P(X)=1/(1+e^(-(%s+%s*X)))" % (theta0, theta1))
#P=1/(1+np.exp^-(theta0+theta1*X))


print("Lets plot it!")

xvals=np.linspace(0,6,100)
print(xvals)
yvals=[]
for hh in xvals:
    print(1.0 / (1 + np.exp(-(theta0+theta1*hh))))
    yvals.append((1.0 / (1 + np.exp(-(theta0+theta1*hh)))))

plt.title("Logistic regression and plotting of the model with the data points",fontsize=14)
plt.ylabel("Probability",fontsize=14)
plt.xlabel("X-Value",fontsize=14)
plt.scatter(X[:,1],Y)
plt.plot(xvals,yvals,'r')
plt.show()
