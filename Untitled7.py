#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.neighbors import KNeighborsClassifier


# In[2]:


X1=[7,7,3,1]
Y1=[7,4,4,4]
target=["BAD","BAD","GOOD","GOOD"]


# In[6]:


f=list(zip(X1,Y1))


# In[7]:


print(zip(X1,Y1))


# In[8]:


knn=KNeighborsClassifier(n_neighbors=3)


# In[10]:


knn.fit(f,target)


# In[13]:


knn.predict([[8,7]])


# In[ ]:




