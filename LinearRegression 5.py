# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 19:33:29 2022

@author: Chris
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:28:28 2022

@author: Chris
"""
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.formula.api as sm
from scipy import stats
import tensorflow as tf

%matplotlib inline

#Data preparation
file = pd.read_csv(r"C:\Users\Chris\Desktop\DTU\3. Semester\02450 - Introduction to Machine Learning and Data Mining\Projekt 2\heart_data.txt")

#Normalizing data
cols = ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 'alcohol', 'age']
data = pd.DataFrame([file[col] for col in cols]).T
famhist = [1 if val == 'Present' else 0 for val in file['famhist']]
data['famhist'] = famhist
data = pd.DataFrame(zscore(data, ddof=1))
cols.append('famhist')
data.columns = cols

results = pd.DataFrame(file['chd'])

#Make data a torch tensor
test = data.tobacco
X = data.drop('tobacco', axis = 1)

x = torch.tensor(X.values, dtype = torch.float)
y = torch.tensor(test.values, dtype = torch.float)
y = y.view(y.shape[0],1)

X_train, X_test, y_train, y_test = train_test_split(x, y)

#Model
n_features = len(cols) - 1
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

#Loss & Optimizer
loss_function = nn.MSELoss()
learning_rate = 0.01

wd = 0.001 #Use this if not looping

step = []
training_loss = []
testing_loss = []

for wd in np.arange(0,0.5, 0.005):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = wd)
    
    #Training
    epochs = 100
    for epoch in range(epochs):
        #Forwards pass + loss
        y_predicted = model(X_train.float())
        loss = loss_function(y_predicted, y_train)
        
        
        #Backwards pass
        loss.backward()
        
        #update
        optimizer.step()
        
        optimizer.zero_grad()
        
        if (epoch+1) % 100 == 0:
            print(f'epoch: {epoch+1}, lambda = {wd:.3f}, loss = {loss.item():.4f}')
    
    
    #Testing
    test_predicted = model(X_test.float())
    test_loss = loss_function(test_predicted, y_test)
    test_loss = test_loss.detach()

    step.append(wd)
    training_loss.append(loss.item())
    testing_loss.append(test_loss)

#plot
predicted = model(X_test).detach()
plt.plot(y_test, 'ro')
plt.plot(predicted, 'b')
plt.show()

plt.plot(step, training_loss, 'g', label ='training loss')
plt.plot(step, testing_loss, 'r', label = 'testing loss')
plt.legend()
plt.xlabel('Regularization parameter')
plt.ylabel('Loss')
plt.show()




    
