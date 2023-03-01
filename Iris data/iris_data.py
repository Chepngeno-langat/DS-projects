#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


iris_data = pd.read_csv('iris.csv', on_bad_lines='skip')
iris_data


# In[3]:


iris_data.info()


# In[4]:


iris_data.describe


# In[5]:


variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

iris_data = iris_data.replace(['Setosa', 'Versicolor' , 'Virginica'], [0, 1, 2])


# In[6]:


X = iris_data.drop('variety', axis=1)
y = iris_data['variety']


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[8]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)


# In[9]:


from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, logmodel.predict(X_test)))


# In[10]:


# Function for classification based on inputs
def classify(a, b, c, d):
    arr = np.array([a, b, c, d]) # Convert to numpy array
    arr = arr.astype(np.float64) # Change the data type to float
    query = arr.reshape(1, -1) # Reshape the array
    prediction = variety_mappings[logreg.predict(query)[0]] # Retrieve from dictionary
    return prediction

