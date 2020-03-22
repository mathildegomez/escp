#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import collections
from collections import Counter

from imblearn.over_sampling import SMOTE

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


# In[3]:


train=pd.read_csv('train.csv')
print(train.head())


# In[4]:


features = train
total = features.isnull().sum()
print(total)


# In[5]:


X=train.drop(["label","purchaseTime"], axis=1)
y=train['label']


# In[6]:


X = X.drop(["C1","C10","C4","N4","N8","hour","visitTime"], axis=1,errors="ignore")
print(X.head())


# In[7]:


X.shape


# In[8]:


x_train, x_test, y_train, y_test=train_test_split(X,y,test_size=0.33, stratify=y)


# In[9]:


print('Original dataset shape %s' % Counter(y_train))
sm=SMOTE(random_state=123, sampling_strategy=0.1)
x_res, y_res=sm.fit_resample(x_train,y_train)


# In[10]:


logit=LogisticRegression()
logit.fit(x_res,y_res)
logit_pred = logit.predict(x_test)


# In[13]:


print(classification_report(y_test,logit_pred))


# In[14]:


confusion_matrix(logit_pred,y_test)


# In[15]:


x_test["prob"] = logit_pred
final = x_test[["id", "prob"]]
final.describe()


# In[16]:


final.to_csv('gomezassignment23', index=False)


# In[ ]:




