#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:43:21 2019

@author: parth
"""

######### Import libraries #########
import math
import matplotlib.pyplot as plt
import numpy as np

########## Prediction funciton ##########
def prediction(A, weights):
    return A[0]*weights[0]+A[1]*weights[1]+A[2]*weights[2]+A[3]*weights[3]+A[4]*weights[4]+A[5]*weights[5]

########## Cost function ##########
def cost(X,Y, weights):
    err = 0
    for i in range(len(Y)):
        k = 0
        for j in range(100):
            k = prediction(X[i], weights)
        t = k - Y[i]
        err += (0.5/len(Y))*math.pow(t,2)
    return err

########## Gradient Prediction function ##########
def grad_prediction(x,Y, weights, k):
    ret = 0
    for i in range(len(Y)):
        ret += (prediction(x[i], weights) - Y[i])
    return ret


########## Gradient Descent function ##########
def grad_des(X,Y,weights, learning_rate = 0.2e-8):
    temp = weights
    for i in range(len(weights)):
        temp[i] = weights[i] - ((learning_rate/len(Y))*(grad_prediction(X, Y, weights, i)))
    for i in range(len(weights)):
        weights[i] = temp[i]

    return weights

def check():
    a = input("Enter the values for minutes spent studying per week : ")
    b = input("Enter the values for ounces of beer consumed per week : ")
    a = float(a)
    b = float(b)
    if( a == 0 and b == 0):
        print("\nEnd\n")
        return
    A = [1,a,b, a*b, a*a, b*b]
    q = prediction(A, weights)
    print("Predicted Semester Grade Point Average is: ",round(q , 2) )
    check()

######### Initialization #########
#X = [[],[],[],[],[],[]]
X_test = [[],[],[],[],[],[]]
Y = []

######### Read a file #########
file = open("GPAData.txt")
print(file)
rows = int(file.readline())

for line in file:
    a,b,c = line.strip().split('\t')
    n = 400
    p = 100
    X = np.random.rand(n,p)
    Y = np.random.rand(n,1)
    '''
    X[0].append(float(1))
    X[1].append(rand)
    X[2].append(float(b))
    X[3].append(float(a)*float(b))
    X[4].append(float(a)**2)
    X[5].append(float(b)**2)
    
    
    Y.append(float(c))
    
######### Splitting the data #########
l = int(rows*0.7)
for i in range(6):
    X_test[i] = X[i][l:300]
for i in range(6):
    X[i] = X[i][0:l]


Y_test = Y[l:300]
Y = Y[0:l]
'''
######### Transposing #########
X = np.asarray(X).T

Y = np.asarray(Y)
X_test = np.asarray(X_test).T
Y_test = np.asarray(Y_test)

######### Wieghts Initialized #########
weights = [0.0,0.0,0.0,0.0,0.0,0.0]
weights = np.asarray(weights)

costs = []
iteration = []
for i in range(10000):
    costs.append(cost(X,Y,weights))
    grad_des(X,Y, weights, learning_rate = 0.2e-8)
    iteration.append(i)

print("Intital value of cost function for the train set",cost(X,Y,weights))

######### Plot the graph #########
plt.plot(iteration[:10000], costs[:10000])
plt.xlabel(" Number of Iterations")
plt.ylabel("Cost Function J")
#plt.savefig("plot.png")
plt.show()


print("\nFinal Weight values are: \n",weights)

print("\nCost function for the test set is: ",cost(X_test,Y_test,weights))

check()




