# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 20:11:08 2019

@author: Parth
"""

#Import libraries for array objects and plotting
import csv
from matplotlib import pyplot as plt

#Initialization 
sl_setosa = []
pl_setosa = []
sl_versicolor = []
pl_versicolor = []
sl_virginica = []
pl_virginica = []

#Open the file for read
with open('_IrisData.txt', 'r') as csv_file:
    lines = csv.reader(csv_file, delimiter='	')
    
    #Get the filtered data
    for line in lines:
        if(line[4]=="setosa"):
            sl_setosa.append(float(line[0]))
            pl_setosa.append(float(line[2]))
        
        if(line[4]=="versicolor"):
            sl_versicolor.append(float(line[0]))
            pl_versicolor.append(float(line[2]))
            
        if(line[4]=="virginica"):
            sl_virginica.append(float(line[0]))
            pl_virginica.append(float(line[2]))
    
    #Plot the data        
    plt.scatter(sl_setosa, pl_setosa, color='red', marker='*', label='setosa')
    plt.scatter(sl_versicolor, pl_versicolor, color='orange', marker='s', label='versicolor')
    plt.scatter(sl_virginica,pl_virginica, color='purple', marker='d', label='virginica')
    plt.xkcd()
    
    #Adding labels and title     
    plt.xlabel('Sepal length')
    plt.ylabel('Petal length')
    plt.title('Sepal length x Petal length')

    plt.legend()

    plt.tight_layout()
    
    #Save the image and display
    plt.savefig('panchal_parthnilesh_MyPlot.png')
    plt.show()