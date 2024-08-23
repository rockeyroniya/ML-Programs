#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay
df=pd.read_csv("/home/roniya/Downloads/IRIS.csv")


# In[49]:


df


# In[50]:


df.head()


# In[51]:


df.tail()


# In[52]:


df.shape


# In[53]:


df.columns


# In[54]:


df.isna().sum()


# In[55]:


df.dtypes


# In[56]:


x=df.iloc[:,:-1]


# In[57]:


y=df.iloc[:,-1]


# In[58]:


x.ndim


# In[59]:


y.ndim


# In[60]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)


# In[61]:


scalar=StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test=scalar.transform(x_test)


# In[62]:


knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
y_train


# In[63]:


y_pred


# In[69]:


mat=confusion_matrix(y_test,y_pred)
label=["Iris-versicolor","Iris-virginica","Iris-setosa"]
dis=ConfusionMatrixDisplay(mat,display_labels=label)
dis.plot()
print(mat)


# In[46]:


score=accuracy_score(y_test,y_pred)
score


# In[ ]:





# In[ ]:




