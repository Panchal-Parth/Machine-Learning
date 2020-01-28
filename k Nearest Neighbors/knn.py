#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:57:39 2019

@author: parth
"""

import random
import math
import operator
import warnings
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		print(distances[x][0])
		neighbors.append(distances[x][0])
	return neighbors
 
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def init(i):
    trainingSet = []
    tn = 0
    tp = 0
    fp = 0
    fn = 0
    error = 0
    fcount = 0
    first = i*48
    last = (i+1)*48
    errors = []
    global cv
    global data
    cv.extend(np.array(data[first:last]))
    trainingSet = data[:first]+data[last:]
    
    for qw in range(15):
        predictions=[]
        e1 = []
        accuracies=[]
        tn = 0
        tp = 0
        fp = 0
        fn = 0
        k = (qw+1)*2-1
        for b in range(len(test)):
            nearone = getNeighbors(trainingSet, test[b], k)
            result = getResponse(nearone)
            predictions.append(result)        
        
        for i in range(len(test)):
            error = 0
            if test[i][-1] == 0 and predictions[i] == 0:
                tn = tn + 1
            elif test[i][-1] == 1 and predictions[i] == 1:
                tp = tp + 1
            elif test[i][-1] == 1 and predictions[i] == 0:
                fp = fp + 1
            elif test[i][-1] == 0 and predictions[i] == 1:
                fn = fn + 1
            else:
                print("ERORR")
            error = error + fp+fn           
            
        try:
            accuracy = (tp+tn)/(tp+tn+fp+fn)
        except ZeroDivisionError:
            accuracy = 0
        try:
            precision = tp / (tp+fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = tp / (tp+fn)
        except ZeroDivisionError:
            recall = 0
        try:
            f1 = 2*(1/((1/precision)+(1/recall)))
        except ZeroDivisionError:
            f1 = 0
        errors.append(error)
        accuracies.append(round(accuracy, 2))
    print(errors)
    #print("\n")
    #print("e1 = \n ",e1)
    t_errors.append(errors)
    acc.append(accuracies) 
  
warnings.simplefilter(action='ignore', category=FutureWarning)
file = input("Enter the file name: ")
file = open(file)
print(file)
rows = int(file.readline())
t0_bl = []
t0_fl = []
t1_bl = []
t1_fl = []
t_errors = []
acc = []
label = []
data = []
test = []
cvset = []
datalist = []

for line in file:
    a,b,c= line.strip().split('\t')
    label.append(int(c))
    d1 = line.split()
    if(int(d1[2])==0):
        t0_bl.append(float(d1[0]))
        t0_fl.append(float(d1[1]))
    else:
        t1_bl.append(float(d1[0]))
        t1_fl.append(float(d1[1]))
    d = [float(a),float(b),int(c)]
    data.append(d)
    
random.shuffle(data)
test = data[240:300]
data = data[0:240]
datalist = []
cv = []
for i in range(5):
    #random.shuffle(data)
    init(i)
    
t_errors = np.array(t_errors)
mins=[]
for i in range(15):
    min_errors=0
    for j in range(5):
        min_errors=min_errors+t_errors[j][i] 
    mins.append(min_errors)
print("-------------------------------------------------------------")

acc = np.array(acc)
cvacc = []
tn = tp = fn = fp = error = 0
kset = []
for i in range(15):
    k = (i+1)*2-1
    kset.append(k)
    trainingSet = cv
    predictions = []
    for b in range(len(test)):
        nearone = getNeighbors(trainingSet, test[b], k)
        result = getResponse(nearone)
        predictions.append(result)
    tn = tp = fn = fp = error = 0
    
    # Calculating the tp,tn,fp,fn for accuracy, precision and f1 score
    for i in range(len(test)):
        if test[i][-1] == 0 and predictions[i] == 0:
            tn = tn + 1
        elif test[i][-1] == 1 and predictions[i] == 1:
            tp = tp + 1
        elif test[i][-1] == 1 and predictions[i] == 0:
            fp = fp + 1
        elif test[i][-1] == 0 and predictions[i] == 1:
            fn = fn + 1
        else:
            print("ERORR")
        error = error + fp+fn
        
    # Try catch block for division for zero error    
    try:
        accuracy = (tp+tn)/(tp+tn+fp+fn)
    except ZeroDivisionError:
        accuracy = 0
    try:
        precision = tp / (tp+fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp / (tp+fn)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2*(1/((1/precision)+(1/recall)))
    except ZeroDivisionError:
        f1 = 0
        
    cvacc.append(round(accuracy,2))
    
print(accuracy,precision,recall,f1)
    
#########  Printing The cross validation accuracy  [Uncomment to check the plot]
'''    
val, idx = max((val, idx) for (idx, val) in enumerate(cvacc))
#print(val, idx)
plt.plot(kset, cvacc, '-o')
plt.xlabel("Value of k for KNN")
plt.ylabel("Cross-Validated Accuracy")
plt.savefig("KNN1.png")
plt.show()
'''

#########  Printing The plot for Tiger0 and Tiger1 fish  [Uncomment to check the plot]
'''
plt.scatter(t0_bl, t0_fl, color='red', marker='*', label='Tiger 0')
plt.scatter(t1_bl, t1_fl, color='green', marker='*', label='Tiger 1')
plt.title('Tiger 0 Fish or Tiger 1 Fish')
plt.xlabel('Body length')
plt.ylabel('Fin length')
plt.legend()
plt.tight_layout()
'''

#########  Displaying Confusion matrix
array = [[tn, fp],[fn, tp]]
df_cm = pd.DataFrame(array, index = [i for i in "NY"],
                  columns = [i for i in "NY"])
plt.figure(figsize = (2,2))
sn.heatmap(df_cm, annot=True)
plt.xlabel('Predicted Fish Species')
plt.ylabel('Actual Fish Species')


######## Taking Input from user and predicting the results
while(True):
    ip = input("Enter the body length and fin length : ")
    ip = ip.split()
    #print("ip = ",ip[0])
    new_list = []
    for item in ip:
        new_list.append(float(item))
    if((new_list[0]==0.0 or new_list[0]==0) and (new_list[1]==0.0 or new_list[1]==0)):
        break
    #print("new list =",new_list)
    inp = [new_list]
    #print("inp =",inp)
    for i in range(15):
        k = (i+1)*2-1
        kset.append(k)
        trainingSet = cv
        predictions = []
        for j in range(len(inp)):
            nearone = getNeighbors(trainingSet, inp[j], k)
            result = getResponse(nearone)
            predictions.append(result)
    print("The Predicted values of the fish species is ",int(predictions[0])) 
# Saving the confusion matrix
plt.savefig('ConfusionMatrix.png')
plt.show()



