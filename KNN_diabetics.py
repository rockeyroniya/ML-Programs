#!/usr/bin/env python
# coding: utf-8

# In[90]:


import numpy as np
import pandas as pd


# In[91]:


df=pd.read_csv("/home/roniya/Downloads/diabetes_1.csv")


# In[92]:


df


# In[93]:


df.head


# In[94]:


df.tail


# In[95]:


df.shape


# In[96]:


df.columns


# In[97]:


df.dtypes


# In[98]:


df.isna().sum()


# In[99]:


x=df.iloc[:,:-1]


# In[100]:


y=df.iloc[:,-1]


# In[101]:


y


# In[102]:


x.ndim


# In[103]:


y.ndim


# In[104]:


from sklearn.model_selection import train_test_split


# In[105]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1)


# In[106]:


x_train


# In[107]:


y_train.shape


# In[108]:


x_test.shape


# In[109]:


from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test=scalar.transform(x_test)


# In[110]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_predict=knn.predict(x_test)


# In[111]:


y_predict


# In[112]:


y_test


# In[113]:


from sklearn.metrics import confusion_matrix
mat=confusion_matrix(y_test,y_predict)


# In[114]:


print(mat)


# In[115]:


from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_predict)


# In[116]:


score


# In[117]:


pred=knn.predict(scalar.transform([[3,130,75,35,3,26,0.16,35]]))
pred


# In[ ]:




