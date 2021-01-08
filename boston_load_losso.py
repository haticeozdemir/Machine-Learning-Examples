#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)


# In[9]:


boston.data


# In[8]:


boston.feature_names


# In[11]:


print(boston.DESCR)


# In[14]:


import pandas as pd
bos = pd.DataFrame(boston.data, columns = boston.feature_names)
bos['PRICE'] = boston.target

print(bos.head())


# In[15]:


bos.isnull().sum()


# In[21]:


import numpy as np
X_rooms = bos.RM
y_price = bos.PRICE


X_rooms = np.array(X_rooms).reshape(-1,1)
y_price = np.array(y_price).reshape(-1,1)

print(X_rooms.shape)
print(y_price.shape)


# In[23]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_rooms, y_price, test_size = 0.3, random_state=5)

print(X_train_1.shape)
print(X_test_1.shape)
print(Y_train_1.shape)
print(Y_test_1.shape)


# In[24]:


from sklearn.linear_model import LinearRegression

from sklearn import metrics
from sklearn.metrics import r2_score
from numpy import sqrt

from sklearn.linear_model import Lasso
lr = LinearRegression() 
lr.fit(X_train_1, Y_train_1)

Y_pred = lr.predict(X_test_1)


# In[25]:


lr = Lasso(alpha=5)
lr.fit(X_train_1, Y_train_1)

Y_predRR = lr.predict(X_test_1)


# In[26]:


Lasso_train_score = lr.score(X_train_1, Y_train_1)
Lasso_test_score = lr.score(X_test_1, Y_test_1)
print("Lasso regression train score:", Lasso_train_score)
print("Lasso regression test score:", Lasso_test_score)


# In[28]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




