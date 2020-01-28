
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:00:53 2019

@author: parth
"""

############# Importing the libraries #############
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

############# file input #############
file = input('Enter File Name : ')
if not file.strip():
	file = "FF64.txt"

data= pd.read_csv(file,header = None)
rows= int(data.loc[0,0])
data= pd.read_csv(file,header=None,skiprows=[0], sep='\t')
data = data.reindex(np.random.permutation(data.index))

seperation = int(rows*0.7)
x_all,Y_all = data[:seperation], data[seperation:]
x = x_all.iloc[:, [0,1]].values
Y = x_all.iloc[:, -1].values
x_test = Y_all.iloc[:, [0,1]].values
Y_test = Y_all.iloc[:, -1].values

x1 = x[:,0]
x2 = x[:,1]
xt_x1 = x_test[:,0]
xt_x2 = x_test[:,1]

iter = 1000
lr = 0.5e-3

weight0= 0.0
weight1 = 0.0
weight2 = 0.0
weight3 = 0.0
weight4 = 0.0

############# Calculate the mean and standardize #############
def mean_c(x1):
	return sum(x1) / len(x1)

def standardize(x1):
	result = np.empty(len(x1))
	variances = np.linspace(1,len(x1),len(x1))
	mean_x1 = mean_c(x1)
	for i in range(len(x1)):
		variances[i] = (x1[i]-mean_x1)**2
	stdev = np.sqrt(sum(variances)/(len(x1)))
	for i in range(len(x1)):
		result[i] = (x1[i] - mean_x1)/stdev
	return result, mean_x1, stdev

x1,mean_train_x1,std_train_x1 = standardize(x1)
x2,mean_train_x2,std_train_x2 = standardize(x2)
x = x1,x2
x = np.transpose(np.asarray(x))

############# hypothesis fuction #############
def hxx(w0,w1,w2,w3,w4,x1,x2):
	hx = np.empty(len(x1))
	lx = np.empty(len(x1))
	for i in range(len(x1)):
		hx[i] = w0+(w3*x1[i])+(w4*x2[i])+(w1*(x1[i]**2))+(w2*(x2[i]**2))#+(w3*x1[i]*x2[i])
		lx[i] = 1/(1+(np.exp((-1)*(hx[i]))))
	return lx

############# cost function #############
def cost(w0,w1,w2,w3,w4,x1,x2,Y,lx):
	rows_count = len(Y)
	costs = 0.0
	#lx = hxx(w0,w1,w2,w3,w4,w5,x1,x2)
	small_val = (1e-5)
	for i in range(len(Y)):
		#costs +=  Y[i]*np.log(lx[i]+small_val) + (1-Y[i])*np.log(1-lx[i]+small_val)
		costs +=  (-1)*(Y[i]*(np.log(lx[i])+small_val) + (1-Y[i])*(np.log(1-lx[i]+small_val)))
	costs1 = (1/rows_count)*costs
	return costs1

############# gradient function #############
def gradient_des(w0, w1, w2,w3,w4,x1,x2, Y, lr, iter):
	J_history = list()
	iterarr = list()
	w02, w12, w22, w32,w42 = w0, w1, w2, w3,w4
	costs=0.0
	print('Weights at start ',w0,' ',w1,' ',w2, ' ',w3,' ',w4)
	hx = hxx(w0,w1,w2,w3,w4,x1,x2)
	costs = cost(w0,w1,w2,w3,w4,x1,x2,Y,hx)
	print('Initial cost ',costs)
	J_history.append(costs)
	iterarr.append(0)
	for j in range(1,iter):
		for i in range(len(x1)):
			w0 = w0 - np.sum(lr*(hx[i]-Y[i]))
			w1 = w1 - np.sum(lr*((hx[i]-Y[i])*(x1[i]**2)))
			w2 = w2 - np.sum(lr*((hx[i]-Y[i])*(x2[i]**2)))
			w3 = w3 - np.sum(lr*((hx[i]-Y[i])*(x1[i])))
			w4 = w4 - np.sum(lr*((hx[i]-Y[i])*(x2[i])))
			
		hx = hxx(w0,w1,w2,w3,w4,x1,x2)
		costs = cost(w0,w1,w2,w3,w4,x1,x2,Y,hx)

		J_history.append(costs)
		iterarr.append(j)
	w02, w12, w22, w32,w42 = w0, w1, w2, w3,w4
	return w02, w12, w22,w32,w42,J_history,iterarr

weight0_n,weight1_n,weight2_n,weight3_n,weight4_n,J_history_n, iterarr_n = gradient_des(weight0, weight1, weight2,weight3,weight4,x1,x2, Y, lr, iter)

print('\nFinal value of J', J_history_n[-1])
plt.plot(iterarr_n,J_history_n)
plt.xlabel("Number of iterations")
plt.ylabel("Value of Cost")
plt.show()

def predict(w0_n,w1_n,w2_n,w3_n,w4_n,x1,x2):
	#print('w0_n,w1_n,w2_n,w4_n,w5_n',w0_n,w1_n,w2_n,w3_n,w4_n,w5_n)
	class_type = np.empty(len(x1))
	hx_prmt = hxx(w0_n,w1_n,w2_n,w3_n,w4_n,x1,x2)
	#print(hx_prmt)
	for i in range(len(hx_prmt)):
		if (hx_prmt[i]>=0.5):
			class_type[i] = 1
		else:
			class_type[i] = 0
	return class_type

############# confusion matrix #############
def c_mat(Y,x1,x2,w0_n,w1_n,w2_n,w3_n,w4_n):
	class_type = predict(w0_n,w1_n,w2_n,w3_n,w4_n,x1,x2)
	nonmatch_count = 0
	tp = 0
	tn = 0
	fp = 0
	fn = 0

	for i in range(len(Y)):
		#print('float(Y[i]) , float(class_type[i])',float(Y[i]) , float(class_type[i]))
		if(float(Y[i]) == float(class_type[i])):
			if((float(Y[i])==1) and (float(class_type[i])==1)):
				tp = tp+1
			elif((float(Y[i])==0) and (float(class_type[i])==0)):
				tn = tn+1
		if(float(Y[i]) != float(class_type[i])):
			nonmatch_count = nonmatch_count +1
			if((float(Y[i])==1) and (float(class_type[i])==0)):
				fp = fp+1
			elif((float(Y[i])==0) and (float(class_type[i])==1)):
				fn = fn+1
	print('TP:',tp,'TN:',tn,'FP:',fp,'FN:',fn,'\n')
	return tp,tn,fp,fn

def e_count(Y,x1,x2,w0_n,w1_n,w2_n,w3_n,w4_n):
	error_count = 0
	accuracy = 0
	precision = 0
	accuracy_p = 0
	recall = 0
	F1_score = 0

	tp,tn,fp,fn = c_mat(Y,x1,x2,w0_n,w1_n,w2_n,w3_n,w4_n)

	error_count = fp+fn
	accuracy_p = (1-(error_count/(fp+fn+tp+tn)))*100
	accuracy = (tp+tn)/(tp+tn+fp+fn)
	if ((tp+fp)== 0):
		precision = 0.000001
	else:
		precision = (tp/(tp+fp))
	if ((tp+fn)== 0):
		recall = 0.000001
	else:
		recall = (tp/(tp+fn))

	if(precision ==0 and recall == 0 ):
		precision = 0.000001
		recall = 0.000001
	F1_score = 2*((precision*recall)/(precision+recall))
	print('error_count',error_count,'accuracy',accuracy,'accuracy_p',accuracy_p,'precision',precision,'recall',recall,'F1_score',F1_score)
	return error_count,accuracy,accuracy_p,precision,recall,F1_score

############# input #############
def inp(w0,w1,w2,w3_n,w4_n,mean_x1,mean_x2,stdev_x1,stdev_x2):


	while True:
		testval_x = float(input('Enter length of body(in cm):'))
		testval_y = float(input('Enter length of dorsal fin(in cm):'))
		if(testval_x==0 and testval_y == 0):
			break
		else:
			xt_x1 = testval_x
			xt_x2 = testval_y
            
			xt_x1 = (xt_x1 - mean_x1)/stdev_x1
			xt_x2 = (xt_x2 - mean_x2)/stdev_x2
            
			xt_x1 = np.asarray(xt_x1)* np.ones(1)
			xt_x2 = np.asarray(xt_x1)* np.ones(1)

			pred_prmt = predict(w0_n,w1_n,w2_n,w3_n,w4_n,xt_x1,xt_x2)
			print('Type predicted:',int(pred_prmt))



xt_x1_nonstd = xt_x1
xt_x2_nonstd = xt_x2
xt_x1,mean,std = standardize(xt_x1_nonstd)
xt_x2,mean,std = standardize(xt_x2_nonstd)

hx_test = hxx(w0_n,w1_n,w2_n,w3_n,w4_n,xt_x1,xt_x2)

cost_test = cost(w0_n,w1_n,w2_n,w3_n,w4_n,xt_x1,xt_x2,Y_test,hx_test)
print('\ncost on test data set',cost_test)
e_count(Y_test,xt_x1,xt_x2,w0_n,w1_n,w2_n,w3_n,w4_n)
print('\nweight0',w0_n,'\nweight1',w1_n,'\nweight2',w2_n,'\nweight3',w3_n,'\nweight4',w4_n)
inp(w0,w1,w2,w3_n,w4_n,mean_train_x1,mean_train_x2,std_train_x1,std_train_x2)
