# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:41:09 2020

@author: orkun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def GetLabel(train_data, X_test, dist_order=2, k=3):
    '''return the labels of the k closest point to X'''
    dist = []
    X_train = train_data.iloc[:,0:2].values
    y_train = train_data.iloc[:,2]

    for i in range(0, X_train.shape[0]):
        dist.append(np.linalg.norm(X_train[i]-X_test, ord=dist_order))

    dist_df = pd.DataFrame({'distance':dist, 'class_label':y_train})
    k_labels = dist_df.sort_values(by = 'distance', ascending=True).iloc[1:k+1]

    return(k_labels.class_label.value_counts().index[0])
    
#data
df = pd.read_csv('./iris_data.txt', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'flower_type']
df.drop(['sepal_width', 'petal_length'], axis=1, inplace=True)

# first 30 samples of each flower are train data
train_data = df.iloc[pd.np.r_[0:30, 50:80, 100:130]]
test_data = df.iloc[pd.np.r_[30:50, 80:100, 130:150]]

#graphs
sns.scatterplot(x='sepal_length', y='petal_width', hue='flower_type', data=df)

X_test = test_data.iloc[:,0:2]
y_test = test_data.iloc[:,2]

#nneighbour
k = 15
#k = int(input())

# 1: for manhattan distance, 2:for euclidean distance
distance = 1
#distance = int(input())

y_pred = []

for i in range(0,X_test.shape[0]):
    y_pred.append(GetLabel(train_data, X_test.iloc[i,:], distance, k))

#Accuracy
print(sum(y_test == y_pred) / len(y_pred))

#Error Count
print(sum(y_test != y_pred))