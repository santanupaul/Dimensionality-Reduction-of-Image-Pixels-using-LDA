
# coding: utf-8

# In[2]:

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# In[3]:

# Change the Current Path
os.chdir('/Users/santanupaul/Documents/Personal/Masters in Analytics/UConn/Study Related/Python/Project/fer2013')

get_ipython().system('pwd')


# In[4]:

# Import the pixel matrix 
df = pd.read_csv('fer2013.csv')

print(df.head())

df = df[(df['emotion'] == 3) | (df['emotion'] == 4)] # 3 happy, and 4 Sad

# Split the pixel columns to form different fields for each pixel
df = pd.concat([df[['emotion']], df['pixels'].str.split(" ", expand = True)], axis = 1)
print(df.head())


# In[5]:

# Train Test Split
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

X = train.iloc[:, 1:].values #Store as numpy array
Y = train.iloc[:, 0].values
X = X.astype(int)

X_test = test.iloc[:, 1:].values #Store as numpy array
Y_test = test.iloc[:, 0].values
X_test = X_test.astype(int)


# In[6]:

# Calculate the mean vectors for each class

# Covariance Matrix
def cov(x):
    mean_vec = np.mean(x, axis = 0)

    return (x - mean_vec).T.dot(x - mean_vec)
    
classes = train.emotion.unique() #Unique classes

# Within Class Variation Sw
Sw = np.zeros([X.shape[1], X.shape[1]])
for i in range(len(classes)):
    Sw += cov(train[train.emotion == classes[i]].iloc[:, 1:].values.astype(int))

# Between Class Variation
# St = Sw + Sb
# Sb = St - Sw
St = cov(train.iloc[:, 1:].values.astype(int))

Sb = St - Sw


# In[10]:

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

# Create a list of Eigenvector and Eigenvalue tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the eigenvalue, eigenvector pair from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# As expected all eigen values except the first one is 0 since number of classes here is 2

# Now project the original data into new feature subspace
X_nd = X.dot(eig_pairs[0][1]).reshape(X.shape[0], 1)
print(X_nd.shape)


# In[25]:

# Support Vector Machine Classifier

# Run SVM
# print('Train SVM...')
svc = SVC()
svc.fit(X_nd, Y)

print('Train Score: \n', svc.score(X_nd, Y))
# 78% Train accuracy

# Project the test data into new feature subspace
X_nd_test = X_test.dot(eig_pairs[0][1]).reshape(X_test.shape[0], 1)

print('\n\nTest Score: \n', svc.score(X_nd_test, Y_test))
# 68% test accuracy


# In[ ]:




# In[ ]:



