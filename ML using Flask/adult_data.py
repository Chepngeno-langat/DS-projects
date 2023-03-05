#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd


# In[28]:


adult_data = pd.read_csv('https://raw.githubusercontent.com/hoshangk/machine_learning_model_using_flask_web_framework/master/adult.csv', on_bad_lines='skip')
adult_data


# In[29]:


adult_data.isnull().sum()


# In[30]:


adult_data = adult_data.drop(['fnlwgt', 'educational-num'], axis=1)


# In[31]:


adult_data.replace(['Divorced', 'Married-AF-spouse',
            'Married-civ-spouse', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Widowed'],
           ['divorced', 'married', 'married', 'married',
            'not married', 'not married', 'not married'], inplace=True)


# In[32]:


from sklearn import preprocessing

 
category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                'relationship', 'gender', 'native-country', 'income']
labelEncoder = preprocessing.LabelEncoder()
 
mapping_dict = {}
for col in category_col:
    adult_data[col] = labelEncoder.fit_transform(adult_data[col])
 
    le_name_mapping = dict(zip(labelEncoder.classes_,
                               labelEncoder.transform(labelEncoder.classes_)))
 
    mapping_dict[col] = le_name_mapping
print(mapping_dict)


# In[34]:


from sklearn.model_selection import train_test_split
X = adult_data.drop('income', axis=1)
y = adult_data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[42]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# In[43]:


from sklearn.metrics import accuracy_score
print("Train accuracy...", accuracy_score(y_train, model.predict(X_train)))
print("Test accuracy...", accuracy_score(y_test, model.predict(X_test)))

