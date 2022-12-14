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

#Model
n_features = len(cols) - 1
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

#Loss & Optimizer
loss_function = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 1)

#Training
epochs = 1000
for epoch in range(epochs):
    #Forwards pass + loss
    y_predicted = model(x.float())
    loss = loss_function(y_predicted, y)
    
    
    #Backwards pass
    loss.backward()
    
    #update
    optimizer.step()
    
    optimizer.zero_grad()
    
    if (epoch+1) % 100 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
        
#plot
predicted = model(x).detach()
plt.plot(y, 'ro')
plt.plot(predicted, 'b')
    
